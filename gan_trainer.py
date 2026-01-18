# gan_trainer.py
import torch
import torch.nn as nn
from torch.optim import Adam
import tqdm

from utils.util import transform_past_predict_batch
from nets.gan_cnn import CGANGenerator, CGANCritic


def _gradient_penalty(critic, real, fake, cond, F_dim: int, device: str):
    """
    WGAN-GP gradient penalty on interpolates.
    """
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device = device)
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)
    score_hat = critic(x_hat, cond, F_dim = F_dim)  # [B]

    grad = torch.autograd.grad(
        outputs = score_hat.sum(),
        inputs = x_hat,
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]  # [B, 2, F, T]

    grad = grad.view(B, -1)
    gp = ((grad.norm(2, dim = 1) - 1) ** 2).mean()
    return gp


class GANTrainer:
    def __init__(self, train_dataloader, val_dataloader, config, **kwargs):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        self.n_levels = config.n_levels
        self.past_window = config.past_window
        self.predict_window = config.predict_window

        self.n_epochs = config.n_epochs
        self.lr_g = config.lr_g
        self.lr_d = config.lr_d
        self.weight_decay = config.weight_decay
        self.z_dim = config.z_dim
        self.base_channels = config.base_channels

        self.n_critic = config.n_critic
        self.lambda_gp = config.lambda_gp

        self.drop_probability = config.drop_probability
        self.clip_gradient = config.clip_gradient
        self.early_stop_patience = config.early_stop_patience
        self.early_stop_min_delta = config.early_stop_min_delta

        self.gan_model_saving_path = config.gan_model_saving_path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # dimensions
        self.F = 2 * self.n_levels
        self.T = self.predict_window

    def _process_batch(self, past_batch, predict_batch, trend_batch, volatility_batch, liquidity_batch, imb_batch, past_time_batch, predict_time_batch):
        
        transformed_past_batch, transformed_predict_batch = transform_past_predict_batch(past_batch, predict_batch, self.n_levels)

        # ---------- target x : [B, 2, F, predict_window] ----------
        x_price = transformed_predict_batch[:, :, :, 0:1]  # [B, T, F, 1]
        x_price = x_price * 100

        x_volume = transformed_predict_batch[:, :, :, -1:]
        x_volume = torch.sqrt(x_volume) / 15

        x = torch.concat([x_price, x_volume], dim = -1).permute(0, 3, 2, 1).contiguous()
        x = x.to(self.device)

        # ---------- conditions cond ----------
        past_price_cond = transformed_past_batch[:, :, :, 0:1].permute(0, 3, 2, 1).contiguous()
        past_price_cond = past_price_cond * 100

        past_volume_cond = transformed_past_batch[:, :, :, -1:].permute(0, 3, 2, 1).contiguous()
        past_volume_cond = torch.sqrt(past_volume_cond) / 15

        trend_cond = trend_batch
        trend_cond = torch.where(trend_cond >= 0, torch.sqrt(trend_cond), -torch.sqrt(-trend_cond)) * 10

        volatility_cond = volatility_batch
        volatility_cond = torch.sqrt(volatility_cond) * 10

        liquidity_cond = liquidity_batch / 2
        liquidity_cond = torch.sqrt(liquidity_cond) / 15

        imb_cond = imb_batch
        time_cond = predict_time_batch

        cond = (
            past_price_cond.to(self.device),
            past_volume_cond.to(self.device),
            trend_cond.to(self.device),
            volatility_cond.to(self.device),
            liquidity_cond.to(self.device),
            imb_cond.to(self.device),
            time_cond.to(self.device),
        )

        # classifier-free dropout
        if self.drop_probability > 0 and torch.rand(1).item() < self.drop_probability:
            cond = tuple(torch.zeros_like(item) for item in cond)

        return x, cond

    def train(self):
        self.G = CGANGenerator(z_dim = self.z_dim, base_channels = self.base_channels, out_channels = 2).to(self.device)

        self.D = CGANCritic(base_channels = self.base_channels, in_channels = 2).to(self.device)

        opt_g = Adam(self.G.parameters(), lr = self.lr_g, weight_decay = self.weight_decay)
        opt_d = Adam(self.D.parameters(), lr = self.lr_d, weight_decay = self.weight_decay)

        scaler = torch.amp.GradScaler()

        best_val = float("inf")

        early_stop_counter = 0

        loop = tqdm.trange(self.n_epochs)
        for epoch in loop:
            # ---------------- train ----------------
            self.G.train()
            self.D.train()

            d_loss_running = 0.0
            g_loss_running = 0.0
            n_items = 0

            for it, batch in enumerate(self.train_dataloader):
                (past_batch, predict_batch, trend_batch, volatility_batch,
                 liquidity_batch, imb_batch, past_time_batch, predict_time_batch) = batch

                real, cond = self._process_batch(
                    past_batch, predict_batch, trend_batch, volatility_batch,
                    liquidity_batch, imb_batch, past_time_batch, predict_time_batch
                )
                B = real.size(0)
                n_items += B

                # ---- train critic n_critic steps ----
                for _ in range(self.n_critic):
                    opt_d.zero_grad(set_to_none = True)
                    z = torch.randn(B, self.z_dim, device = self.device)

                    with torch.amp.autocast(device_type = self.device):
                        fake = self.G(z, cond, F_dim = self.F, T_dim = self.T).detach()
                        d_real = self.D(real, cond, F_dim = self.F)
                        d_fake = self.D(fake, cond, F_dim = self.F)
                    
                    gp = _gradient_penalty(self.D, real, fake, cond, F_dim = self.F, device = self.device)
                    d_loss = (d_fake.mean() - d_real.mean()) + self.lambda_gp * gp

                    if torch.isnan(d_loss) or torch.isinf(d_loss):
                        continue

                    scaler.scale(d_loss).backward()
                    if self.clip_gradient is not None:
                        scaler.unscale_(opt_d)
                        nn.utils.clip_grad_norm_(self.D.parameters(), self.clip_gradient)
                    scaler.step(opt_d)
                    scaler.update()

                # ---- train generator ----
                opt_g.zero_grad(set_to_none = True)
                z = torch.randn(B, self.z_dim, device = self.device)

                with torch.amp.autocast(device_type = self.device):
                    fake = self.G(z, cond, F_dim = self.F, T_dim = self.T)
                    g_loss = -self.D(fake, cond, F_dim = self.F).mean()

                if torch.isnan(g_loss) or torch.isinf(g_loss):
                    continue

                scaler.scale(g_loss).backward()
                if self.clip_gradient is not None:
                    scaler.unscale_(opt_g)
                    nn.utils.clip_grad_norm_(self.G.parameters(), self.clip_gradient)
                scaler.step(opt_g)
                scaler.update()

                d_loss_running += d_loss.item() * B
                g_loss_running += g_loss.item() * B

            train_d = d_loss_running / max(n_items, 1)
            train_g = g_loss_running / max(n_items, 1)

            # ---------------- val (use critic gap as proxy) ----------------
            self.G.eval()
            self.D.eval()
            with torch.no_grad():
                val_proxy = 0.0
                val_items = 0
                for batch in self.val_dataloader:
                    (past_batch, predict_batch, trend_batch, volatility_batch,
                     liquidity_batch, imb_batch, past_time_batch, predict_time_batch) = batch

                    real, cond = self._process_batch(
                        past_batch, predict_batch, trend_batch, volatility_batch,
                        liquidity_batch, imb_batch, past_time_batch, predict_time_batch
                    )
                    B = real.size(0)
                    z = torch.randn(B, self.z_dim, device = self.device)
                    fake = self.G(z, cond, F_dim = self.F, T_dim = self.T)

                    d_real = self.D(real, cond, F_dim = self.F).mean()
                    d_fake = self.D(fake, cond, F_dim = self.F).mean()
                    proxy = (d_fake - d_real).abs().item()
                    val_proxy += proxy * B
                    val_items += B

                val_proxy = val_proxy / max(val_items, 1)

            loop.set_description(f"Epoch {epoch} | Train D: {train_d:.5f} | Train G: {train_g:.5f} | ValProxy: {val_proxy:.5f}")

            # early stop + save
            if best_val - val_proxy > self.early_stop_min_delta:
                best_val = val_proxy
                early_stop_counter = 0
                torch.save({"G": self.G.state_dict(), "D": self.D.state_dict()}, self.gan_model_saving_path)
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch}, best val proxy = {best_val}")
                break
