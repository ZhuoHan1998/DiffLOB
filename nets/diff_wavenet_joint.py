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
    def __init__(self, channels, cond_channels, dilation):
        super().__init__()
        self.filter_conv = nn.Conv2d(channels, channels, kernel_size = (3, 3),
                                     padding = (dilation, dilation), dilation = (dilation, dilation))
        self.gate_conv   = nn.Conv2d(channels, channels, kernel_size = (3, 3),
                                     padding = (dilation, dilation), dilation = (dilation, dilation))
        self.time_film   = TimeFiLM(emb_dim = 128, n_channels = channels)
        self.cond_film   = CondFiLM(cond_channels = cond_channels, n_channels = channels)
        self.cond_proj   = nn.Conv2d(cond_channels, channels, kernel_size = 1)

        self.residual_proj = nn.Conv2d(channels, channels, kernel_size = 1)
        self.skip_proj     = nn.Conv2d(channels, channels, kernel_size = 1)

    def forward(self, x, cond, t_emb):
        B, C, F, T = x.shape

        x = self.time_film(x, t_emb)  # [B, base_channels, F, T]
        
        cond_proj = self.cond_proj(cond)  # [B, base_channels, F, T]
        x = self.cond_film(x, cond_proj)
        
        filter_out = torch.tanh(self.filter_conv(x))  # [B,base_channels,F,T] 
        gate_out   = torch.sigmoid(self.gate_conv(x)) # [B,base_channels,F,T] 
        h = filter_out * gate_out                     # [B,base_channels,F,T] 

        residual = self.residual_proj(h)  # [B,base_channels,F,T] 
        skip     = self.skip_proj(h)      # [B,base_channels,F,T] 

        x = x + residual
        return x, skip

class FeatureAttention(nn.Module):
    def __init__(self, channels, num_heads = 8, dropout = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim = channels, num_heads = num_heads, 
                                          dropout = dropout, batch_first = True)

    def forward(self, x):
        B, C, F, T = x.shape
        # reshape to (B * T, F, C)
        x_t = x.permute(0, 3, 2, 1).reshape(B * T, F, C)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        # reshape back to (B, C, F, T)
        attn_out = attn_out.reshape(B, T, F, C).permute(0, 3, 2, 1)
        return attn_out
    
class FeatureProjector(nn.Module):
    """ Projection + Normalization to map feature channels to a unified latent space. """
    def __init__(self, in_channels, out_channels, num_groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        # GroupNorm can force normalization
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))
    
    
class MultiModalAdapter(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.proj = FeatureProjector(1, base_channels)

    def forward(self, x, cond):
        
        # 1. TARGET Projection (X)
        x_price_emb = self.proj(x[:, 0:1, :, :]) 
        x_vol_emb   = self.proj(x[:, 1:2, :, :])
        x_emb = x_price_emb + x_vol_emb

        # 2. CONDITION Projection & Alignment (C)
        past_price_cond_emb = self.proj(cond[:, 0:1, :, :]) 
        past_volume_cond_emb = self.proj(cond[:, 1:2, :, :]) 
        trend_cond_emb = self.proj(cond[:, 2:3, :, :]) 
        volatility_cond_emb = self.proj(cond[:, 3:4, :, :]) 
        liquidity_cond_emb = self.proj(cond[:, 4:5, :, :]) 
        oi_cond_emb = self.proj(cond[:, 5:6, :, :]) 
        time_cond_emb = self.proj(cond[:, 6:7, :, :]) 
        
        cond_emb = past_price_cond_emb + past_volume_cond_emb + trend_cond_emb + \
                   volatility_cond_emb + liquidity_cond_emb + oi_cond_emb + time_cond_emb
        
        return x_emb, cond_emb


class WaveNetJoint(nn.Module):
    def __init__(self, input_dim = 2, cond_dim = 7, output_dim = 2, base_channels = 256, 
                 num_layers = 32, n_features = 20, num_heads = 32, num_steps = 100):
        super().__init__()
        # self.adapter = MultiModalAdapter(base_channels = base_channels)
        
        self.target_proj = nn.Conv2d(input_dim, base_channels, kernel_size = 1)
        self.cond_proj   = nn.Conv2d(cond_dim,  base_channels, kernel_size = 1)

        self.time_embed = DiffusionEmbedding(num_steps = num_steps, embedding_dim = 128)
        
        self.dim_emb = nn.Embedding(n_features, base_channels)
        
        self.feat_attn = FeatureAttention(base_channels, num_heads = num_heads)

        # gated WaveNet blocks
        self.blocks = nn.ModuleList()
        base_cycle = [1, 2, 4, 8, 16]  
        dilations = [base_cycle[i % len(base_cycle)] for i in range(num_layers)]
        for d in dilations:
            self.blocks.append(GatedWaveNetBlock(base_channels, base_channels, dilation = d))

        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size = 1),
            nn.ReLU(),
            nn.Conv2d(base_channels, output_dim, kernel_size = 1)
        )
        
    def forward(self, x, t, cond):
        
        # x_proj, cond_proj = self.adapter(x.float(), cond.float())  # [B, base_channels, F, T]
        
        x_proj = self.target_proj(x.float())  # [B, base_channels, F, T]
        cond_proj = self.cond_proj(cond.float())  # [B, base_channels, F, T]

        # add feature-dimension embedding
        B, C, F, T = x.shape
        dims = torch.arange(F, device = x.device)  # [F]
        dims_emb = self.dim_emb(dims)  # [F, base_channels]
        dims_emb = dims_emb.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # [1, base_channels, F, 1]
        
        x_proj = x_proj + dims_emb

        # time embedding        
        t_emb = self.time_embed(t) # [B, embedding_dim]

        # cross-feature attention
        x_proj = self.feat_attn(x_proj) # [B, base_channels, F, T]

        # gated convolutions
        skip_total = 0
        for block in self.blocks:
            x_proj, skip = block(x_proj, cond_proj, t_emb)
            skip_total = skip_total + skip # torch.Size([64, 256, 20, 32])
        
        out = self.output_proj(skip_total)
        
        return out # torch.Size([64, 2, 20, 32])



