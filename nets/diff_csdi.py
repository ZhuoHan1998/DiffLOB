import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_torch_trans(heads=8, layers=1, channels=64):
    """
    Create a standard Transformer Encoder using PyTorch's implementation.
    """
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads,
        dim_feedforward=64, activation="gelu",
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_ch, out_ch, kernel_size):
    """
    1D conv with Kaiming initialization.
    """
    layer = nn.Conv1d(in_ch, out_ch, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    """
    Sinusoidal + MLP projection for diffusion timesteps.
    """
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False
        )
        self.proj1 = nn.Linear(embedding_dim, projection_dim)
        self.proj2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.proj1(x)
        x = F.silu(x)
        x = self.proj2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1)
        freqs = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * freqs
        return torch.cat([table.sin(), table.cos()], dim=1)


class diff_CSDI(nn.Module):
    """
    A conditional diffusion model network with per-feature dimension conditioning.
    """
    def __init__(self, inputdim = 2, side_dim = 7, channels = 64, num_steps = 100, 
                 diffusion_embedding_dim = 128, nheads = 8, layers = 4, features = 20, ):
        """
        Args:
            config (dict): must now contain either
                - config["features"] = number of features
            inputdim (int): Number of input channels.
        """
        super().__init__()
        self.inputdim = inputdim
        self.channels = channels
        # look for 'features' or fallback to 'K'
        self.K = features
        if self.K is None:
            raise ValueError("config must specify 'features' (or 'K') = number of features in your input")

        # Diffusion embedding for timestep conditioning
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps = num_steps,
            embedding_dim = diffusion_embedding_dim
        )

        # Dimension embedding: one learnable bias vector per feature index
        self.dim_emb = nn.Embedding(self.K, self.channels)

        # Input / output projections
        self.input_projection   = Conv1d_with_init(self.inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, self.inputdim, 1)
        nn.init.zeros_(self.output_projection2.weight)

        # Stack of ResidualBlocks
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                side_dim = side_dim,
                channels = channels,
                diffusion_embedding_dim = diffusion_embedding_dim,
                nheads = nheads,
            )
            for _ in range(layers)
        ])
        

    def _pack_cond_tuple_to_tensor(self, cond_tuple, K: int, L: int, device):
        """
        Trainer cond tuple:
          (past_price [B,1,K,L], past_volume [B,1,K,L],
           trend [B] or [B,1], volatility [B] or [B,1],
           liquidity [B,L], imb [B,L], time [B,L])

        Return:
          cond_info [B, side_dim(=7), K, L]
        """
        (past_price, past_vol, trend, vol, liq, imb, time) = cond_tuple

        past_price = past_price.to(device)
        past_vol   = past_vol.to(device)

        # trend/vol: [B] or [B,1] -> [B,1,K,L]
        if trend.dim() == 1:
            trend = trend[:, None]
        if vol.dim() == 1:
            vol = vol[:, None]
        trend = trend.to(device).view(-1, 1, 1, 1).expand(-1, 1, K, L)
        vol   = vol.to(device).view(-1, 1, 1, 1).expand(-1, 1, K, L)

        # seq: [B,L] -> [B,1,K,L]
        def _seq_to_map(x):
            if x.dim() == 3:  # [B,1,L]
                x = x.squeeze(1)
            x = x.to(device).view(-1, 1, 1, L).expand(-1, 1, K, L)
            return x

        liq  = _seq_to_map(liq)
        imb  = _seq_to_map(imb)
        time = _seq_to_map(time)

        cond_info = torch.cat([past_price, past_vol, trend, vol, liq, imb, time], dim=1)  # [B,7,K,L]
        return cond_info


    def forward(self, x, t, cond, enable_motion: bool = False, enable_control: bool = False):
        """
        Match diffusion_trainer.py calling convention:
          net(x, t, cond, enable_motion, enable_control)

        Args:
            x:    (B, inputdim, K, L)
            t:    (B,) or (B,1)
            cond: tuple of 7 tensors (Trainer) OR packed tensor (B, side_dim, K, L)
        """
        # normalize t: [B]
        if t.dim() == 2 and t.size(-1) == 1:
            t = t.squeeze(-1)
        t = torch.round(t).long().to(x.device)

        B, _, K, L = x.shape
        assert K == self.K, f"Expected K={self.K}, but got input with K={K}"

        # pack cond if tuple/list
        if isinstance(cond, (tuple, list)):
            cond_info = self._pack_cond_tuple_to_tensor(cond, K=K, L=L, device=x.device)
        else:
            cond_info = cond.to(x.device)

        # sanity shape
        assert cond_info.dim() == 4 and cond_info.size(0) == B and cond_info.size(2) == K and cond_info.size(3) == L, \
            "cond_info must be (B, side_dim, K, L) and match x in (B,K,L)"

        # ---- original logic below (unchanged) ----
        x = x.reshape(B, self.inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(t)

        skip_connections = []
        for layer in self.residual_layers:
            x, skip = layer(x, cond_info, diffusion_emb)
            skip_connections.append(skip)

        x = torch.stack(skip_connections).sum(0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)

        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, self.inputdim, K, L)
        return x

    
class ResidualBlock(nn.Module):
    """
    A single residual block for the diffusion model.

    Each block incorporates time-step conditioning via diffusion embeddings, applies
    time and feature transformers, and fuses conditional information.
    """
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        """
        Args:
            side_dim (int): Number of channels for the side (conditional) input.
            channels (int): Number of channels in the main branch.
            diffusion_embedding_dim (int): Dimensionality of the diffusion embedding.
            nheads (int): Number of attention heads for the transformer.
            is_linear (bool): Whether to use linear attention transformers.
        """
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # Use standard PyTorch transformer
        self.time_layer = get_torch_trans(heads = nheads, layers = 1, channels = channels)
        self.feature_layer = get_torch_trans(heads = nheads, layers = 1, channels = channels)

    def forward_time(self, y, base_shape):
        """
        Process the input along the temporal dimension using a transformer.
        
        Args:
            y (Tensor): Input tensor of shape (B, channels, K*L).
            base_shape (tuple): The original shape (B, channels, K, L).
        
        Returns:
            Tensor: Transformed tensor with the same reshaped dimensions.
        """
        B, channel, K, L = base_shape
        # If there is only one time step, skip transformation
        if L == 1:
            return y
        # Reshape to group time dimension for transformer: (B*K, channel, L)
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)


        # Standard transformer expects shape (L, B*K, channel)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        # Reshape back to (B, channels, K*L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        """
        Process the input along the feature dimension using a transformer.
        
        Args:
            y (Tensor): Input tensor of shape (B, channels, K*L).
            base_shape (tuple): The original shape (B, channels, K, L).
        
        Returns:
            Tensor: Transformed tensor with updated feature interactions.
        """
        B, channel, K, L = base_shape
        # If there is only one feature, skip transformation
        if K == 1:
            return y
        # Reshape to group feature dimension for transformer: (B*L, channel, K)
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)

        # Standard transformer expects shape (K, B*L, channel)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        # Reshape back to (B, channels, K*L)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        """
        Forward pass for the residual block.

        Args:
            x (Tensor): Main input tensor of shape (B, channels, K, L).
            cond_info (Tensor): Conditional input tensor of shape (B, cond_dim, K, L).
            diffusion_emb (Tensor): Diffusion embedding tensor of shape (B, diffusion_embedding_dim).
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - The residual output added to the input.
                - The skip connection tensor.
        """
        
        B, channel, K, L = x.shape
        base_shape = x.shape
        # Flatten spatial dimensions: (B, channels, K*L)
        x = x.reshape(B, channel, K * L)

        # Project diffusion embedding and add it to x
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # Shape: (B, channel, 1)
        y = x + diffusion_emb

        # Process time and feature dimensions separately via transformers
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # Shape: (B, channel, K*L)
        y = self.mid_projection(y)  # Shape: (B, 2*channel, K*L)

        # Process conditional information
        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # Shape: (B, 2*channel, K*L)
        y = y + cond_info

        # Gated activation: split into two parts and apply non-linearities
        gate, filter = torch.chunk(y, 2, dim = 1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # Shape: (B, channel, K*L)
        y = self.output_projection(y)

        # Split into residual and skip connections
        residual, skip = torch.chunk(y, 2, dim = 1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        # Combine input with residual and return along with the skip connection
        return (x + residual) / math.sqrt(2.0), skip

