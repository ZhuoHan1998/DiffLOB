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
        liquidity_cond:   [B,T]      (ignored in this ultra-simple baseline)
        imb_cond:         [B,T]      (ignored)
        time_cond:        [B,T]      (ignored)
    )

    Output:
      cond_map: [B, Cc, F, T]  (concat of past_map + global_map)
    """
    def __init__(self, cond_channels: int = 32):
        super().__init__()
        self.cond_channels = cond_channels

        # past map: [B,2,F,T] -> [B,Cc,F,T]
        self.past_net = nn.Sequential(
            nn.Conv2d(2, cond_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cond_channels, cond_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # global scalars: [trend, vol] -> [B,Cc]
        self.global_net = nn.Sequential(
            nn.Linear(2, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
        )

        # fuse to keep channels compact
        self.fuse = nn.Sequential(
            nn.Conv2d(cond_channels * 2, cond_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(cond_channels, cond_channels, kernel_size=1),
        )

    def forward(self, cond, F_dim: int, T_dim: int):
        past_price, past_volume, trend, vol, *_ = cond

        past = torch.cat([past_price, past_volume], dim=1)     # [B,2,F,T]
        past_map = self.past_net(past)                         # [B,Cc,F,T]

        g = torch.stack([trend, vol], dim=-1)                  # [B,2]
        g = self.global_net(g)                                 # [B,Cc]
        global_map = g.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, F_dim, T_dim)  # [B,Cc,F,T]

        cond_map = self.fuse(torch.cat([past_map, global_map], dim=1))  # [B,Cc,F,T]
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


class CGANGenerator(nn.Module):

    def __init__(self, z_dim: int = 128, base_channels: int = 128, cond_channels: int = 32, out_channels: int = 2):
        super().__init__()
        self.z_dim = z_dim
        self.base_channels = base_channels
        self.cond_enc = CondEncoder(cond_channels=cond_channels)

        # z -> [B, base_channels, 1, 1]
        self.z_proj = nn.Linear(z_dim, base_channels)

        # input channels: z_map(base) + cond_map(cond_channels)
        self.in_conv = nn.Conv2d(base_channels + cond_channels, base_channels, kernel_size = 1)

        self.body = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
        )

        self.out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size = 1),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size = 1),
        )

    def forward(self, z, cond, F_dim: int, T_dim: int):
        cond_map = self.cond_enc(cond, F_dim = F_dim, T_dim = T_dim)  # [B, C, F, T]

        z_h = self.z_proj(z).unsqueeze(-1).unsqueeze(-1)           # [B, C, 1, 1]
        z_map = z_h.expand(-1, -1, F_dim, T_dim)                   # [B, C, F, T]

        h = self.in_conv(torch.cat([z_map, cond_map], dim = 1))
        h = self.body(h)
        return self.out(h)


class CGANCritic(nn.Module):

    def __init__(self, base_channels: int = 128, cond_channels: int = 32, in_channels: int = 2):
        super().__init__()
        self.cond_enc = CondEncoder(cond_channels = cond_channels)

        # input: x(in_channels) + cond_map(cond_channels)
        self.in_conv = nn.Conv2d(in_channels + cond_channels, base_channels, kernel_size = 1)

        self.body = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
        )

        self.head = nn.Sequential(
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out = nn.Linear(base_channels, 1)

    def forward(self, x, cond, F_dim: int):
        # infer T from x
        T_dim = x.shape[-1]
        cond_map = self.cond_enc(cond, F_dim = F_dim, T_dim = T_dim)   # [B,Cc,F,T]
        h = self.in_conv(torch.cat([x, cond_map], dim = 1))           # [B,C,F,T]
        h = self.body(h)
        h = self.head(h).flatten(1)                                 # [B,C]
        return self.out(h).squeeze(-1)                              # [B]
