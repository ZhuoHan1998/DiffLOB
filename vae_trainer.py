import torch
import torch.nn as nn
from torch.optim import Adam
import tqdm

from utils.util import transform_past_predict_batch
from nets.vae_cnn import ConditionalVAE


class VAETrainer:
    def __init__(self, train_dataloader, val_dataloader, config, **kwargs):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        self.n_levels = config.n_levels
        self.past_window = config.past_window
        self.predict_window = config.predict_window

        # --- VAE hyperparams ---
        self.n_epochs = config.n_epochs
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.clip_gradient = config.clip_gradient

        self.z_dim = config.z_dim
        self.base_channels = config.base_channels
        self.cond_channels = config.cond_channels

        # KL settings
        self.beta_kl = config.beta_kl
        self.kl_anneal = config.kl_anneal
        self.kl_anneal_epochs = config.kl_anneal_epochs

        self.recon_loss_type = config.recon_loss_type

        self.early_stop_patience = config.early_stop_patience
        self.early_stop_min_delta = config.early_stop_min_delta

        self.vae_model_saving_path = config.vae_model_saving_path

        self.drop_probability = config.drop_probability

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.F = 2 * self.n_levels

        if self.recon_loss_type == "mse":
            self.recon_fn = nn.MSELoss(reduction = "mean")
        else:
            self.recon_fn = nn.L1Loss(reduction = "mean")

    def _process_batch(self, past_batch, predict_batch, trend_batch, volatility_batch, liquidity_batch, imb_batch, past_time_batch, predict_time_batch):
        """
        Returns:
          x:   [B,2,F,T]
          cond tuple (7 items)
        """
        transformed_past_batch, transformed_predict_batch = transform_past_predict_batch(
            past_batch, predict_batch, self.n_levels
        )

        # --- x ---
        x_price = transformed_predict_batch[:, :, :, 0:1] * 100
        x_volume = torch.sqrt(transformed_predict_batch[:, :, :, -1:]) / 15
        x = torch.concat([x_price, x_volume], dim=-1).permute(0, 3, 2, 1).contiguous().to(self.device)

        # --- cond ---
        past_price_cond = transformed_past_batch[:, :, :, 0:1].permute(0, 3, 2, 1).contiguous() * 100
        past_volume_cond = torch.sqrt(transformed_past_batch[:, :, :, -1:]).permute(0, 3, 2, 1).contiguous() / 15

        trend_cond = torch.where(trend_batch >= 0, torch.sqrt(trend_batch), -torch.sqrt(-trend_batch)) * 10
        volatility_cond = torch.sqrt(volatility_batch) * 10
        liquidity_cond = torch.sqrt((liquidity_batch / 2)) / 15

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

        # optional cond dropout
        if self.drop_probability > 0 and torch.rand(1).item() < self.drop_probability:
            cond = tuple(torch.zeros_like(item) for item in cond)

        return x, cond

    @staticmethod
    def _kl_divergence(mu, logvar):
        # KL(q(z|x)||p(z)) for diagonal Gaussian
        # = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()

    def _beta_at_epoch(self, epoch: int):
        if not self.kl_anneal:
            return self.beta_kl
        # linear warmup
        warm = max(1, self.kl_anneal_epochs)
        return self.beta_kl * min(1.0, float(epoch + 1) / float(warm))

    def train(self):
        self.vae = ConditionalVAE(
            z_dim = self.z_dim,
            base_channels = self.base_channels,
            cond_channels = self.cond_channels,
            x_channels = 2,
        ).to(self.device)

        opt = Adam(self.vae.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        scaler = torch.amp.GradScaler()

        best_val = float("inf")
        early_stop_counter = 0

        loop = tqdm.trange(self.n_epochs)
        for epoch in loop:
            beta = self._beta_at_epoch(epoch)

            # ---------------- train ----------------
            self.vae.train()
            train_loss = 0.0
            train_items = 0

            for batch in self.train_dataloader:
                (past_batch, predict_batch, trend_batch, volatility_batch,
                 liquidity_batch, imb_batch, past_time_batch, predict_time_batch) = batch

                x, cond = self._process_batch(
                    past_batch, predict_batch, trend_batch, volatility_batch,
                    liquidity_batch, imb_batch, past_time_batch, predict_time_batch
                )
                B = x.size(0)
                train_items += B

                opt.zero_grad(set_to_none = True)
                with torch.amp.autocast(device_type = "cuda"):
                    x_hat, mu, logvar = self.vae(x, cond)
                    recon = self.recon_fn(x_hat, x)
                    kl = self._kl_divergence(mu, logvar)
                    loss = recon + beta * kl

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                scaler.scale(loss).backward()
                if self.clip_gradient is not None:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(self.vae.parameters(), self.clip_gradient)
                scaler.step(opt)
                scaler.update()

                train_loss += loss.item() * B

            train_loss = train_loss / max(train_items, 1)

            # ---------------- val ----------------
            self.vae.eval()
            val_loss = 0.0
            val_items = 0
            with torch.no_grad():
                for batch in self.val_dataloader:
                    (past_batch, predict_batch, trend_batch, volatility_batch,
                     liquidity_batch, imb_batch, past_time_batch, predict_time_batch) = batch

                    x, cond = self._process_batch(
                        past_batch, predict_batch, trend_batch, volatility_batch,
                        liquidity_batch, imb_batch, past_time_batch, predict_time_batch
                    )
                    B = x.size(0)
                    val_items += B

                    with torch.amp.autocast(device_type="cuda"):
                        x_hat, mu, logvar = self.vae(x, cond)
                        recon = self.recon_fn(x_hat, x)
                        kl = self._kl_divergence(mu, logvar)
                        loss = recon + beta * kl

                    val_loss += loss.item() * B

            val_loss = val_loss / max(val_items, 1)

            loop.set_description(
                f"Epoch {epoch} | beta={beta:.3f} | Train={train_loss:.6f} | Val={val_loss:.6f}"
            )

            # save / early stop
            if best_val - val_loss > self.early_stop_min_delta:
                best_val = val_loss
                early_stop_counter = 0
                torch.save(self.vae.state_dict(), self.vae_model_saving_path)
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch}, best val = {best_val}")
                break
