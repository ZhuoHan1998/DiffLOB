import torch
import torch.nn as nn
import torch.nn.functional as F


class CondEncoder(nn.Module):
    """
    cond = (
      past_price_cond:  [B,1,F,T]
      past_volume_cond: [B,1,F,T]
      trend_cond:       [B]
      volatility_cond:  [B]
      liquidity_cond:   [B,T]
      imb_cond:         [B,T]
      time_cond:        [B,T]
    )
    """
    def __init__(self, cond_channels: int = 64, seq_in_dim: int = 3):
        super().__init__()
        self.cond_channels = cond_channels

        # past map: [B,2,F,T] -> [B,C,F,T]
        self.past_net = nn.Sequential(
            nn.Conv2d(2, cond_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cond_channels, cond_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # seq (liq, imb, time): [B,T,3] -> [B,C,T] -> [B,C,1,T] -> [B,C,F,T]
        self.seq_proj = nn.Linear(seq_in_dim, cond_channels)
        self.seq_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
        )

        # global scalars (trend, vol): [B,2] -> [B,C] -> broadcast [B,C,F,T]
        self.global_mlp = nn.Sequential(
            nn.Linear(2, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
        )

        # fuse: [past, seq, global] -> [B,C,F,T]
        self.fuse = nn.Sequential(
            nn.Conv2d(cond_channels * 3, cond_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(cond_channels, cond_channels, kernel_size=1),
        )

    def forward(self, cond, F_dim: int, T_dim: int):
        past_price, past_volume, trend, vol, liq, imb, time = cond

        past = torch.cat([past_price, past_volume], dim=1)       # [B,2,F,T]
        past_map = self.past_net(past)                           # [B,C,F,T]

        seq = torch.stack([liq, imb, time], dim=-1)              # [B,T,3]
        seq = self.seq_proj(seq).permute(0, 2, 1).contiguous()   # [B,C,T]
        seq = self.seq_conv(seq)                                 # [B,C,T]
        seq_map = seq.unsqueeze(2).expand(-1, -1, F_dim, -1)     # [B,C,F,T]

        g = torch.stack([trend, vol], dim=-1)                    # [B,2]
        g = self.global_mlp(g)                                   # [B,C]
        g_map = g.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, F_dim, T_dim)  # [B,C,F,T]

        cond_map = self.fuse(torch.cat([past_map, seq_map, g_map], dim=1))  # [B,C,F,T]
        return cond_map


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.skip = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class ConditionalVAE(nn.Module):
    """
    Encoder:  (x, cond_map) -> mu, logvar  (latent dim z_dim)
    Decoder:  (z, cond_map) -> x_hat
    """
    def __init__(
        self,
        z_dim: int = 64,
        base_channels: int = 128,
        cond_channels: int = 64,
        x_channels: int = 2,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.base_channels = base_channels
        self.cond_channels = cond_channels

        self.cond_enc = CondEncoder(cond_channels=cond_channels)

        # --- Encoder ---
        self.enc_in = nn.Conv2d(x_channels + cond_channels, base_channels, kernel_size=1)
        self.enc_body = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
        )
        self.enc_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.enc_mu = nn.Linear(base_channels, z_dim)
        self.enc_logvar = nn.Linear(base_channels, z_dim)

        # --- Decoder ---
        self.z_proj = nn.Linear(z_dim, base_channels)
        self.dec_in = nn.Conv2d(base_channels + cond_channels, base_channels, kernel_size=1)
        self.dec_body = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
        )
        self.dec_out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, x_channels, kernel_size=1),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        # z = mu + std * eps
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, cond):
        B, _, F_dim, T_dim = x.shape
        cond_map = self.cond_enc(cond, F_dim=F_dim, T_dim=T_dim)                 # [B,C,F,T]
        h = self.enc_in(torch.cat([x, cond_map], dim=1))
        h = self.enc_body(h)
        h = self.enc_pool(h).flatten(1)                                          # [B,C]
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def decode(self, z, cond, F_dim: int, T_dim: int):
        cond_map = self.cond_enc(cond, F_dim=F_dim, T_dim=T_dim)                 # [B,C,F,T]
        z_h = self.z_proj(z).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, F_dim, T_dim)  # [B,C,F,T]
        h = self.dec_in(torch.cat([z_h, cond_map], dim=1))
        h = self.dec_body(h)
        x_hat = self.dec_out(h)                                                  # [B,2,F,T]
        return x_hat

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, cond, F_dim=x.shape[-2], T_dim=x.shape[-1])
        return x_hat, mu, logvar
