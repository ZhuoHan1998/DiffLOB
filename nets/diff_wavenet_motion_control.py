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
        gamma, beta = self.linear(t_emb).chunk(2, dim = -1) # [B, base_channels]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) # [B, base_channels, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)   # [B, base_channels, 1, 1]
        return x * (1 + gamma) + beta # [B, base_channels, F, T]

    
class GlobalFiLM(nn.Module):
    """
    x:        [B, base_channels, F, T]
    cond_emb: [B, embedding_dim]  
    """
    def __init__(self, emb_dim, n_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(emb_dim, n_channels, kernel_size = 1),
            nn.SiLU(),
            nn.Conv2d(n_channels, 2 * n_channels, kernel_size = 1),
        )

    def forward(self, x, cond_emb):
        # [B, D] -> [B, D, 1, 1]
        cond = cond_emb.unsqueeze(-1).unsqueeze(-1)

        # Generate FiLM params
        gamma_beta = self.net(cond)                # [B, 2C, 1, 1]
        gamma, beta = gamma_beta.chunk(2, dim = 1) # each [B, C, 1, 1]

        # Broadcast over (F, T)
        return x * (1 + gamma) + beta
    
    
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
        self.filter_conv = nn.Conv2d(channels, channels, kernel_size = (3, 3),
                                     padding = (dilation, dilation), dilation = (dilation, dilation))
        self.gate_conv   = nn.Conv2d(channels, channels, kernel_size = (3, 3),
                                     padding = (dilation, dilation), dilation = (dilation, dilation))
        self.time_film   = TimeFiLM(emb_dim = 128, n_channels = channels)
        self.global_film = GlobalFiLM(emb_dim = 128, n_channels = channels)
        self.cond_film   = CondFiLM(cond_channels = channels, n_channels = channels)
        self.cond_proj   = nn.Conv2d(channels, channels, kernel_size = 1)

        self.residual_proj = nn.Conv2d(channels, channels, kernel_size = 1)
        self.skip_proj     = nn.Conv2d(channels, channels, kernel_size = 1)

    def forward(self, x, cond, global_emb, t_emb):
        B, C, F, T = x.shape
        
        x_input = x
        
        x = self.norm(x)

        x = self.time_film(x, t_emb)  # [B, C, F, T]
        
        x = self.global_film(x, global_emb)

        x = self.cond_film(x, cond)
        
        filter_out = torch.tanh(self.filter_conv(x))  # [B, C, F, T] 
        gate_out   = torch.sigmoid(self.gate_conv(x)) # [B, C, F, T] 
        h = filter_out * gate_out                     # [B, C, F, T] 

        residual = self.residual_proj(h)  # [B, C, F, T] 
        skip     = self.skip_proj(h)      # [B, C, F, T] 

        x = x_input + residual
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
    def __init__(self, channels, emb_dim):
        super().__init__()
        
        self.global_film = GlobalFiLM(emb_dim, channels)
        
        self.zero_conv_in = ZeroConv2d(channels)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.SiLU()
        )
        
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, channels),
            nn.Tanh()
        )
        
        self.zero_conv_out = ZeroConv2d(channels)
        
        self.alpha = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, cond_proj, global_emb):
        '''
        x:          [B, C, F, T]
        cond_proj:  [B, C, F, T]
        global_emb: [B, emb_dim]
        return:     [B, C, F, T]
        ''' 
        
        B, C, F, T = cond_proj.shape
        
        cond_modulated = self.global_film(cond_proj, global_emb)
        
        x_in = x + self.zero_conv_in(cond_modulated)   
        
        x_in = self.encoder(x_in)
        
        bias = self.proj(global_emb).unsqueeze(-1).unsqueeze(-1)
        
        x_in = x_in + bias
        
        control_signal = self.zero_conv_out(x_in)
        
        return control_signal
    
    
class MotionModule(nn.Module):
    """
    x:             [B, C, F, T]
    motion_signal: [B, C, F, T]
    """
    def __init__(self, channels: int, hidden: int = None, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        hidden = hidden if hidden is not None else max(1, channels // 2)

        self.in_proj = nn.Conv2d(channels * 2, channels, kernel_size=1)

        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.proj = nn.Linear(hidden, channels)

        self.to_delta_gate = nn.Conv2d(channels, channels * 2, kernel_size=1)
        
        self.global_film = GlobalFiLM(emb_dim = 128, n_channels = channels)
        self.cond_film   = CondFiLM(cond_channels = channels, n_channels = channels)

        # zero conv in/out
        self.zero_conv_in = ZeroConv2d(channels)
        self.zero_conv_out = ZeroConv2d(channels)

        self.alpha = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, cond_proj, global_emb):
        B, C, Feature, T = x.shape

        # small residual hint
        x_in = x + self.zero_conv_in(x)

        # ---- difference dynamics ----
        dx = x_in[..., 1:] - x_in[..., :-1]   # [B,C,F,T-1]
        dx = F.pad(dx, (1, 0))                # [B,C,F,T]
        feat = torch.cat([x_in, dx], dim=1)   # [B,2C,F,T]
        feat = self.in_proj(feat)             # [B,C,F,T]
        
        # FiLM injection
        feat = self.global_film(feat, global_emb)
        feat = self.cond_film(feat, cond_proj)

        # ---- GRU per-feature ----
        feat_seq = feat.permute(0, 2, 3, 1).contiguous().view(B * Feature, T, C)  # [B*F,T,C]
        h, _ = self.gru(feat_seq)               # [B*F,T,hidden]
        h = self.proj(h)                        # [B*F,T,C]
        h = h.view(B, Feature, T, C).permute(0, 3, 1, 2).contiguous()  # [B,C,F,T]

        # ---- gated delta ----
        dg = self.to_delta_gate(h)              # [B,2C,F,T]
        delta, gate = dg.chunk(2, dim=1)        # [B,C,F,T] each
        gate = torch.sigmoid(gate)
        delta = gate * delta                    # [B,C,F,T]

        # ---- final controlled injection ----
        alpha = torch.sigmoid(self.alpha)       # scalar in (0,1)
        motion_signal = self.zero_conv_out(alpha * delta)  # keep ControlNet-style

        return motion_signal

    
    
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
        # reshape to (B * F, T, C)
        x_t = x.permute(0, 2, 3, 1).contiguous().view(B * F, T, C)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        # reshape back to (B, C, F, T)
        attn_out = attn_out.view(B, F, T, C).permute(0, 3, 1, 2).contiguous()
        return attn_out

    
class ConditionEncoder(nn.Module):
    """
    Encode your heterogeneous conditions into:
      1) cond_map: [B, C, F, T]  (for CondFiLM / additive injection)
      2) global_emb: [B, emb_dim] (to be added into time embedding)

    Expected cond tuple:
      (past_price_cond, past_volume_cond, trend_cond, volatility_cond, liquidity_cond, imb_cond, time_cond)

    Shapes:
      past_price_cond:  [B, 1, F, T]
      past_volume_cond: [B, 1, F, T]
      trend_cond:       [B]
      volatility_cond:  [B]
      liquidity_cond:   [B, T]
      imb_cond:         [B, T]
      time_cond:        [B, T]
      
    Returns:
        cond_map [B, C, F, T] includes past price/volume, liquidity, imb, time;
        global_emb [B, emb_dim] includes trend/vol.
    """
    def __init__(self, base_channels = 256, emb_dim = 128, seq_in_dim = 3):
        super().__init__()
        self.base_channels = base_channels
        self.emb_dim = emb_dim

        # past map encoder: [B,2,F,T] -> [B,C,F,T]
        self.past_in = nn.Conv2d(2, base_channels, kernel_size=1)
        self.past_net = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )

        # seq encoder: [B, T, 3] -> [B, C, 1, T] -> broadcast to [B,C,F,T]
        self.seq_proj = nn.Linear(seq_in_dim, base_channels)
        self.seq_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
        )
        
        # fuse past + seq -> cond_map
        self.fuse = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
        )

        # global scalar encoder: [B,2] -> [B,emb_dim]
        self.global_mlp = nn.Sequential(
            nn.Linear(2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        
    def forward(self, cond, F_dim):
        (past_price, past_volume, trend, vol, liq, imb, time) = cond

        past = torch.cat([past_price, past_volume], dim = 1)  # [B,2,F,T]
        h_past = self.past_net(self.past_in(past))  # [B,C,F,T]

        seq = torch.stack([liq, imb, time], dim = -1)  # [B, T, 3]
        h_seq = self.seq_proj(seq)                  # [B, T, C]
        h_seq = h_seq.permute(0, 2, 1)              # [B, C, T]
        h_seq = self.seq_conv(h_seq)                # [B, C, T]
        h_seq = h_seq.unsqueeze(2)                  # [B, C, 1, T]
        h_seq = h_seq.expand(-1, -1, F_dim, -1)     # [B, C, F, T]

        # ---- fuse to cond_map ----
        cond_map = self.fuse(torch.cat([h_past, h_seq], dim = 1))  # [B, C, F, T]

        # ---- global scalar emb ----
        g = torch.stack([trend, vol], dim = -1)  # [B, 2]
        global_emb = self.global_mlp(g)        # [B, emb_dim]

        return cond_map, global_emb
    

class WaveNetJoint(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 2, base_channels = 128, emb_dim = 128,
                 num_layers = 16, n_features = 20, num_heads = 32, num_steps = 100):
        super().__init__()
        
        self.target_proj = nn.Conv2d(input_dim, base_channels, kernel_size = 1)

        self.time_embed = DiffusionEmbedding(num_steps = num_steps, embedding_dim = emb_dim)
        
        self.cond_encoder = ConditionEncoder(base_channels = base_channels, emb_dim = emb_dim)
        
        self.dim_emb = nn.Embedding(n_features, base_channels)
        
        self.feat_attn = FeatureAttention(base_channels, num_heads = num_heads)

        # gated WaveNet blocks (Spatial + Motion)
        self.blocks = nn.ModuleList()
        self.motion_blocks = nn.ModuleList()
        self.control_blocks = nn.ModuleList()
        
        base_cycle = [1, 2, 4, 8, 16]  
        dilations = [base_cycle[i % len(base_cycle)] for i in range(num_layers)]
        for d in dilations:
            self.blocks.append(GatedWaveNetBlock(base_channels, dilation = d))
            self.motion_blocks.append(MotionModule(channels = base_channels, hidden = base_channels // 4))
            self.control_blocks.append(ControlModule(base_channels, emb_dim))

        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size = 1),
            nn.ReLU(),
            nn.Conv2d(base_channels, output_dim, kernel_size = 1)
        )
        
    def forward(self, x, t, cond, enable_motion:bool, enable_control:bool):
        
        x_proj = self.target_proj(x.float())  # [B, base_channels, F, T]

        # add feature-dimension embedding
        B, C, F, T = x.shape
        dims = torch.arange(F, device = x.device)  # [F]
        dims_emb = self.dim_emb(dims)  # [F, base_channels]
        dims_emb = dims_emb.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # [1, base_channels, F, 1]
        
        x_proj = x_proj + dims_emb

        # cross-feature attention
        # x_proj = self.feat_attn(x_proj) # [B, base_channels, F, T]
        
        # time embedding        
        t_emb = self.time_embed(t) # [B, embedding_dim]
        
        # ---- condition encoding ----
        cond_proj, global_emb = self.cond_encoder(cond, F_dim = F)
        
        skip_total = 0
        for block, motion_block, ctrl_block in zip(self.blocks, self.motion_blocks, self.control_blocks):
             
            # both are before main block    
            block_input = x_proj
            
            if enable_motion:
                motion_signal = motion_block(block_input, cond_proj, global_emb)
                block_input = block_input + motion_signal
                
            if enable_control:
                ctrl_signal = ctrl_block(block_input, cond_proj, global_emb)
                block_input = block_input + ctrl_signal

            # Pass through Main Block
            x_proj, skip = block(block_input, cond_proj, global_emb, t_emb)
            skip_total = skip_total + skip  # torch.Size([B, C, F, T])
        
        out = self.output_proj(skip_total)
        
        return out # torch.Size([B, C, F, T])



