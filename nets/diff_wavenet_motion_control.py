import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionEmbedding(nn.Module):
    """
    Sinusoidal + MLP projection for diffusion timesteps.
    """
    def __init__(self, num_steps, embedding_dim = 128):
        super().__init__()
        embed = self._build_embedding(num_steps, embedding_dim // 2) # [num_steps, embedding_dim]
        self.register_buffer("embedding", embed, persistent = False)
        self.proj1 = nn.Linear(embedding_dim, embedding_dim)
        self.proj2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, diffusion_step):
        """
        diffusion_step: LongTensor of shape [B]
        """
        diffusion_step = diffusion_step.long()        
        
        x = self.embedding[diffusion_step] # [B, embedding_dim]
        x = self.proj1(x)
        x = F.silu(x)
        x = self.proj2(x)
        x = F.silu(x)
        return x  # [B, projection_dim]

    def _build_embedding(self, num_steps, dim):
        """
        num_steps: int
        dim: embedding_dim // 2
        Using sin() and cos() to construct a mapping
        """
        steps = torch.arange(num_steps).unsqueeze(1)             # [num_steps, 1]
        freqs = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0)    # [dim]
        table = steps * freqs.unsqueeze(0)                       # [num_steps, dim]
        emb = torch.cat([table.sin(), table.cos()], dim = 1)     # [num_steps, dim*2]
        return emb


class TimeFiLM(nn.Module):
    """
    Feature-wise Linear Modulation to inject the time embedding
    x:     [B, base_channels, F, T]
    t_emb: [B, embedding_dim]
    """
    def __init__(self, emb_dim, n_channels):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, n_channels * 2),
            nn.SiLU(),
            nn.Linear(n_channels * 2, n_channels * 2)
        )

    def forward(self, x, t_emb):
        gamma, beta = self.linear(t_emb).chunk(2, dim = -1) # [B, embedding_dim]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) # [B, embedding_dim, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)   # [B, embedding_dim, 1, 1]
        return x * (1 + gamma) + beta # [B, base_channels, F, T]


class CondFiLM(nn.Module):
    """
    x:    [B, base_channels, F, T]
    cond: [B, base_channels, F, T]
    """
    def __init__(self, cond_channels, n_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(cond_channels, n_channels, kernel_size = 1),
            nn.SiLU(),
            nn.Conv2d(n_channels, n_channels * 2, kernel_size = 1)
        )

    def forward(self, x, cond):
        gamma, beta = self.fusion(cond).chunk(2, dim = 1) # [B, base_channels, F, T]
        return x * (1 + gamma) + beta # [B, base_channels, F, T]

    
class GatedWaveNetBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups = 8, num_channels = channels, eps = 1e-5)
        self.filter_conv = nn.Conv2d(channels, channels, kernel_size = (3, 1),
                                     padding = (dilation, 0), dilation = (dilation, 1))
        self.gate_conv   = nn.Conv2d(channels, channels, kernel_size = (3, 1),
                                     padding = (dilation, 0), dilation = (dilation, 1))
        self.time_film   = TimeFiLM(emb_dim = 128, n_channels = channels)
        self.cond_film   = CondFiLM(cond_channels = channels, n_channels = channels)
        self.cond_proj   = nn.Conv2d(channels, channels, kernel_size = 1)

        self.residual_proj = nn.Conv2d(channels, channels, kernel_size = 1)
        self.skip_proj     = nn.Conv2d(channels, channels, kernel_size = 1)

    def forward(self, x, cond, t_emb):
        '''
        x:      [B, C, F, T]
        return: [B, C, F, T]
        '''
        B, C, F, T = x.shape
        
        residual_input = x
        
        x = self.norm(x)

        x = self.time_film(x, t_emb)  # [B, C, F, T]
        
        x = self.cond_film(x, cond)   # [B, C, F, T]
        
        filter_out = torch.tanh(self.filter_conv(x))  # [B, C, F, T]
        gate_out   = torch.sigmoid(self.gate_conv(x)) # [B, C, F, T]
        h = filter_out * gate_out                     # [B, C, F, T]

        residual = self.residual_proj(h)  # [B, C, F, T]
        skip     = self.skip_proj(h)      # [B, C, F, T]

        x = residual_input + residual
        return x, skip
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x:       [B, T, d_model]
        return:  [B, T, d_model]
        '''
        return x + self.pe[:, :x.size(1), :]
    
    
class MotionModule(nn.Module):
    def __init__(self, channels, num_heads = 8, dropout = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim = channels, num_heads = num_heads, 
                                          dropout = dropout, batch_first = True)
        
        self.proj_out = nn.Linear(channels, channels)
        
        # ZERO INITIALIZATION (Critical for AnimateDiff)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        
        self.norm = nn.LayerNorm(channels)
        
        self.pos_encoder = PositionalEncoding(d_model = channels, max_len = 128)

    def forward(self, x):
        # x: [B, C, F, T]
        B, C, F, T = x.shape
        
        # Reshape to treat each Price Level as an independent sequence
        # [B, C, F, T] -> [B, F, T, C] -> [B*F, T, C]
        x_in = x.permute(0, 2, 3, 1).reshape(B * F, T, C)
        
        # Position encoding before attention
        x_in = self.pos_encoder(x_in)
        
        # Temporal Attention
        attn_out, _ = self.attn(x_in, x_in, x_in)
        
        # Residual + Zero Init Projection
        x_out = self.proj_out(attn_out)
        x_out = self.norm(x_in + x_out)
        
        # Reshape back to [B, C, F, T]
        x_out = x_out.reshape(B, F, T, C).permute(0, 3, 1, 2)
        
        return x + x_out

    
class AnimateDiffBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.spatial_block = GatedWaveNetBlock(channels, dilation)
        self.motion_module = MotionModule(channels)

    def forward(self, x, cond, t_emb, enable_motion = False):
        '''
        x: [B, C, F, T]
        ''' 
        B, C, F, T = x.shape
        
        # [B, C, F, T] -> [B, T, C, F] -> [B*T, C, F, 1]
        x_spatial = x.permute(0, 3, 1, 2).reshape(B * T, C, F, 1)

        # cond: [B, C, F, T] -> [B*T, C, F, 1]
        cond_spatial = cond.permute(0, 3, 1, 2).reshape(B * T, C, F, 1)
        
        # t_emb: [B, D] -> [B, 1, D] -> [B, T, D] -> [B*T, D]
        t_emb_spatial = t_emb.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        
        x_spatial, skip_spatial = self.spatial_block(x_spatial, cond_spatial, t_emb_spatial)

        # [B*T, C, F, 1] -> [B, T, C, F] -> [B, C, F, T]
        x = x_spatial.reshape(B, T, C, F).permute(0, 2, 3, 1)
        skip = skip_spatial.reshape(B, T, C, F).permute(0, 2, 3, 1)
        
        # --- 2. TEMPORAL PASS ---
        if enable_motion:
            # Motion module handles the internal reshaping [B, C, F, T] -> [B*F, T, C]
            x = self.motion_module(x)
            
        return x, skip

    
class ZeroConv2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 1)
        # Force initialization to zero
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)
    
    
class ControlModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.zero_conv_in = ZeroConv2d(channels)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.SiLU()
        )
        
        self.zero_conv_out = ZeroConv2d(channels)

    def forward(self, x, cond):
        '''
        x:      [B, C, F, T]
        cond:   [B, C, F, T]
        return: [B, C, F, T]
        ''' 
        
        x_in = x + self.zero_conv_in(cond)
        
        x_in = self.encoder(x_in)
        
        control_signal = self.zero_conv_out(x_in)
        
        return self.zero_conv_out(control_signal)
    
    
class FeatureAttention(nn.Module):
    def __init__(self, channels, num_heads = 8, dropout = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim = channels, num_heads = num_heads, 
                                          dropout = dropout, batch_first = True)

    def forward(self, x):
        '''
        x:      [B, C, F, T]
        return: [B, C, F, T]
        '''
        B, C, F, T = x.shape
        # reshape to (B * T, F, C)
        x_t = x.permute(0, 3, 2, 1).reshape(B * T, F, C)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        # reshape back to (B, C, F, T)
        attn_out = attn_out.reshape(B, T, F, C).permute(0, 3, 2, 1)
        return attn_out


class WaveNetJoint(nn.Module):
    def __init__(self, input_dim = 2, cond_dim = 7, output_dim = 2, base_channels = 256, 
                 num_layers = 32, n_features = 20, num_heads = 32, num_steps = 100):
        super().__init__()
        
        self.target_proj = nn.Conv2d(input_dim, base_channels, kernel_size = 1)
        self.cond_proj   = nn.Conv2d(cond_dim,  base_channels, kernel_size = 1)

        self.time_embed = DiffusionEmbedding(num_steps = num_steps, embedding_dim = 128)
        
        self.dim_emb = nn.Embedding(n_features, base_channels)
        
        self.feat_attn = FeatureAttention(base_channels, num_heads = num_heads)

        # gated WaveNet blocks (Spatial + Motion)
        self.blocks = nn.ModuleList()
        # ControlModule blocks
        self.control_blocks = nn.ModuleList()
        
        base_cycle = [1, 2, 4, 8, 16]  
        dilations = [base_cycle[i % len(base_cycle)] for i in range(num_layers)]
        for d in dilations:
            self.blocks.append(AnimateDiffBlock(base_channels, dilation = d))
            self.control_blocks.append(ControlModule(base_channels))

        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size = 1),
            nn.ReLU(),
            nn.Conv2d(base_channels, output_dim, kernel_size = 1)
        )
        
    def forward(self, x, t, cond, enable_motion:bool, enable_control:bool):
        
        x_proj = self.target_proj(x.float())  # [B, C, F, T]
        cond_proj = self.cond_proj(cond.float())  # [B, C, F, T]

        # add feature-dimension embedding
        B, C, F, T = x.shape
        
        dims = torch.arange(F, device = x.device)  # [F]
        dims_emb = self.dim_emb(dims)  # [F, C]
        dims_emb = dims_emb.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # [1, C, F, 1]
        
        x_proj = x_proj + dims_emb  # [B, C, F, T]

        # time embedding        
        t_emb = self.time_embed(t) # [B, embedding_dim]

        x_proj = self.feat_attn(x_proj)  # [B, C, F, T]
        
        skip_total = 0
        for block, ctrl_block in zip(self.blocks, self.control_blocks):
            
            if enable_control:
                ctrl_signal = ctrl_block(x_proj, cond_proj)
                block_input = x_proj + ctrl_signal
            else:
                block_input = x_proj

            # Pass through Main Block
            x_proj, skip = block(block_input, cond_proj, t_emb, enable_motion)
            skip_total = skip_total + skip  # torch.Size([B, C, F, T])
        
        out = self.output_proj(skip_total)
        
        return out # torch.Size([B, C, F, T])



