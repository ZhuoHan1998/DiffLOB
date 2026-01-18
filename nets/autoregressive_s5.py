# nets/autoregressive_s5.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RealS5ZOHLayer(nn.Module):
    """
    u: [B,T,d_model]
    state: [B,G,N]
    y: [B,T,d_model]
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 32,
        groups: int = 8,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        learn_dt: bool = True,
    ):
        super().__init__()
        assert d_model % groups == 0, "d_model must be divisible by groups"
        self.d_model = d_model
        self.state_dim = state_dim
        self.groups = groups
        self.group_dim = d_model // groups

        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.learn_dt = bool(learn_dt)

        # A is always negative: A = -softplus(A_log)  (diagonal, per-group)
        self.A_log = nn.Parameter(torch.randn(groups, state_dim) * 0.02)

        # B, C, D per group
        # B: [G,N,gd]  C: [G,gd,N]  D: [G,gd]
        self.B = nn.Parameter(torch.randn(groups, state_dim, self.group_dim) * 0.02)
        self.C = nn.Parameter(torch.randn(groups, self.group_dim, state_dim) * 0.02)
        self.D = nn.Parameter(torch.zeros(groups, self.group_dim))

        # learnable dt per group (much more stable than per-channel then mean)
        if self.learn_dt:
            self.dt_logit = nn.Parameter(torch.zeros(groups))
        else:
            self.register_buffer("dt_logit", torch.zeros(groups), persistent=False)

    def _A(self):
        return -F.softplus(self.A_log)  # [G,N] strictly negative

    def _dt(self):
        # dt in (dt_min, dt_max), per-group
        sig = torch.sigmoid(self.dt_logit)  # [G]
        return self.dt_min + (self.dt_max - self.dt_min) * sig  # [G]

    @staticmethod
    def _safe_frac(Abar_minus_1: torch.Tensor, A: torch.Tensor, dt: torch.Tensor, eps: float = 1e-6):
        """
        Compute (exp(A dt)-1)/A stably for small |A|.
        Abar_minus_1, A: [G,N], dt: [G,1]
        """
        denom_ok = (A.abs() > eps)
        frac = torch.where(denom_ok, Abar_minus_1 / A, dt.expand_as(A))
        return frac

    def forward(self, u: torch.Tensor, state=None):
        Bsz, T, D = u.shape
        assert D == self.d_model

        if state is None:
            state = u.new_zeros(Bsz, self.groups, self.state_dim)

        A = self._A()                         # [G,N]
        dt = self._dt().to(u.device)          # [G]
        dt_g = dt.view(self.groups, 1)        # [G,1]

        # ZOH discretization for diagonal A (real)
        Adt = A * dt_g                        # [G,N]
        Abar = torch.exp(Adt)                 # [G,N]
        frac = self._safe_frac(Abar - 1.0, A, dt_g)  # [G,N]
        # Bbar = frac[:, :, None] * B
        Bbar = frac.unsqueeze(-1) * self.B    # [G,N,gd]

        # split input into groups: [B,T,G,gd]
        u_g = u.view(Bsz, T, self.groups, self.group_dim)

        ys = []
        s = state                             # [B,G,N]
        for t in range(T):
            u_t = u_g[:, t, :, :]             # [B,G,gd]

            # s <- Abar*s + Bbar*u
            # Abar*s: [B,G,N]
            s = s * Abar.unsqueeze(0) + torch.einsum("bgi,gni->bgn", u_t, Bbar)

            # y = C*s + D*u
            Cs = torch.einsum("bgn,gdn->bgd", s, self.C)         # [B,G,gd]
            y_t = Cs + self.D.unsqueeze(0) * u_t                 # [B,G,gd]
            ys.append(y_t)

        y = torch.stack(ys, dim=1).contiguous().view(Bsz, T, D)
        return y, s

    @torch.no_grad()
    def step(self, u_t: torch.Tensor, state=None):
        Bsz, D = u_t.shape
        assert D == self.d_model

        if state is None:
            state = u_t.new_zeros(Bsz, self.groups, self.state_dim)

        A = self._A()                        # [G,N]
        dt = self._dt().to(u_t.device)       # [G]
        dt_g = dt.view(self.groups, 1)

        Adt = A * dt_g
        Abar = torch.exp(Adt)
        frac = self._safe_frac(Abar - 1.0, A, dt_g)
        Bbar = frac.unsqueeze(-1) * self.B   # [G,N,gd]

        u_g = u_t.view(Bsz, self.groups, self.group_dim)  # [B,G,gd]

        state = state * Abar.unsqueeze(0) + torch.einsum("bgi,gni->bgn", u_g, Bbar)
        Cs = torch.einsum("bgn,gdn->bgd", state, self.C)
        y_g = Cs + self.D.unsqueeze(0) * u_g
        y = y_g.contiguous().view(Bsz, D)
        return y, state


class ChannelMix(nn.Module):
    """
    Lightweight mixing to mimic the 'mixing' part in many SSM stacks.
    Use GLU-style gating for a bit more expressiveness.
    """
    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = expansion * d_model
        self.fc = nn.Linear(d_model, 2 * hidden)
        self.proj = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B,T,D]
        a, b = self.fc(x).chunk(2, dim=-1)
        x = a * torch.sigmoid(b)             # GLU
        x = self.proj(x)
        return self.drop(x)

    def step(self, x_t: torch.Tensor):
        # x_t: [B,D]
        a, b = self.fc(x_t).chunk(2, dim=-1)
        x_t = a * torch.sigmoid(b)
        x_t = self.proj(x_t)
        return self.drop(x_t)


class RealS5Block(nn.Module):
    """
    PreNorm + residual SSM + residual channel-mix FFN (GLU).
    """
    def __init__(
        self,
        d_model: int,
        state_dim: int,
        groups: int,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        learn_dt: bool = True,
        mix_expansion: int = 2,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.s5 = RealS5ZOHLayer(
            d_model=d_model,
            state_dim=state_dim,
            groups=groups,
            dt_min=dt_min,
            dt_max=dt_max,
            learn_dt=learn_dt,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mix = ChannelMix(d_model=d_model, expansion=mix_expansion, dropout=dropout)

    def forward(self, x: torch.Tensor, state=None):
        h = self.ln1(x)
        y, new_state = self.s5(h, state=state)
        x = x + self.drop1(y)

        h2 = self.ln2(x)
        x = x + self.mix(h2)
        return x, new_state

    def step(self, x_t: torch.Tensor, state=None):
        h = self.ln1(x_t)
        y, new_state = self.s5.step(h, state=state)
        x_t = x_t + self.drop1(y)

        h2 = self.ln2(x_t)
        x_t = x_t + self.mix.step(h2)
        return x_t, new_state


class AutoregressiveS5Model(nn.Module):
    """
    Same interface as before:
      forward(prev_x, cond_seq) -> [B,T,x_dim]
      step(prev_x_t, cond_t, caches) -> (x_pred, caches)
    """
    def __init__(
        self,
        x_dim: int,
        cond_dim: int = 5,
        d_model: int = 256,
        n_layers: int = 6,
        state_dim: int = 32,
        dropout: float = 0.0,
        groups: int = 8,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        learn_dt: bool = True,
        mix_expansion: int = 2,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.groups = groups

        self.in_proj = nn.Linear(x_dim + cond_dim, d_model)
        self.blocks = nn.ModuleList([
            RealS5Block(
                d_model=d_model,
                state_dim=state_dim,
                groups=groups,
                dropout=dropout,
                dt_min=dt_min,
                dt_max=dt_max,
                learn_dt=learn_dt,
                mix_expansion=mix_expansion,
            )
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_model, x_dim)

    def forward(self, prev_x: torch.Tensor, cond_seq: torch.Tensor):
        h = self.in_proj(torch.cat([prev_x, cond_seq], dim=-1))  # [B,T,D]
        states = [None] * self.n_layers
        for i, blk in enumerate(self.blocks):
            h, states[i] = blk(h, state=states[i])
        return self.out_proj(h)

    @torch.no_grad()
    def init_cache(self, batch_size: int, device):
        caches = []
        for blk in self.blocks:
            caches.append(torch.zeros(batch_size, blk.s5.groups, blk.s5.state_dim, device=device))
        return caches

    @torch.no_grad()
    def step(self, prev_x_t: torch.Tensor, cond_t: torch.Tensor, caches=None):
        if caches is None:
            caches = self.init_cache(prev_x_t.size(0), prev_x_t.device)

        h = self.in_proj(torch.cat([prev_x_t, cond_t], dim=-1))  # [B,D]
        new_caches = []
        for i, blk in enumerate(self.blocks):
            h, s = blk.step(h, state=caches[i])
            new_caches.append(s)

        x_pred = self.out_proj(h)  # [B,x_dim]
        return x_pred, new_caches
