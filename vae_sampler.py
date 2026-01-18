import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from utils.util import transform_past_predict_batch, transform_sample_batch
from nets.vae_cnn import ConditionalVAE


class VAESampler:
    def __init__(self, sample_dataloader, config, **kwargs):
        self.sample_dataloader = sample_dataloader
        self.config = config

        self.n_levels = config.n_levels
        self.past_window = config.past_window
        self.predict_window = config.predict_window
        self.store_length = config.store_length
        self.sampling_batch_size = config.sampling_batch_size

        # VAE settings
        self.z_dim = config.z_dim
        self.base_channels = config.base_channels
        self.cond_channels = config.cond_channels

        self.vae_model_saving_path = config.vae_model_saving_path
        self.vae_samples_saving_path = config.vae_samples_saving_path

        # sampler knobs (same style)
        self.AR = config.AR
        self.refresh_cycle = config.refresh_cycle

        self.responsive_liquidity = config.responsive_liquidity
        self.responsive_imb = config.responsive_imb
        self.responsive_trend = config.responsive_trend
        self.responsive_volatility = config.responsive_volatility

        self.liquidity_cond_path = config.liquidity_cond_path
        self.imb_cond_path = config.imb_cond_path
        self.trend_cond_path = config.trend_cond_path
        self.volatility_cond_path = config.volatility_cond_path
        self.bin_mode = config.bin_mode

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.F = 2 * self.n_levels

    def get_conditional_pool(self, npy_path, selector, mode = "quantile", q = 5):
        responsive_cond = np.load(npy_path)
        if mode == "quantile":
            bins = pd.qcut(responsive_cond, q = q, labels = False)
        elif mode == "range":
            bins = pd.cut(responsive_cond, bins = q, labels = False)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        target = 0 if selector == "low" else q - 1
        pool = responsive_cond[bins == target]
        return torch.tensor(pool, dtype = torch.float32)

    def sample(self):
        self.vae = ConditionalVAE(
            z_dim = self.z_dim,
            base_channels = self.base_channels,
            cond_channels = self.cond_channels,
            x_channels = 2,
        ).to(self.device)

        ckpt = torch.load(self.vae_model_saving_path, map_location = self.device)
        self.vae.load_state_dict(ckpt, strict = True)
        self.vae.eval()

        # responsive pools
        if self.responsive_liquidity is not None and self.liquidity_cond_path is not None:
            liquidity_pool = self.get_conditional_pool(self.liquidity_cond_path, self.responsive_liquidity, self.bin_mode)
        if self.responsive_imb is not None and self.imb_cond_path is not None:
            imb_pool = self.get_conditional_pool(self.imb_cond_path, self.responsive_imb, self.bin_mode)
        if self.responsive_trend is not None and self.trend_cond_path is not None:
            trend_pool = self.get_conditional_pool(self.trend_cond_path, self.responsive_trend, self.bin_mode)
        if self.responsive_volatility is not None and self.volatility_cond_path is not None:
            volatility_pool = self.get_conditional_pool(self.volatility_cond_path, self.responsive_volatility, self.bin_mode)

        fake_samples = []
        loop = tqdm(self.sample_dataloader, desc = "VAE Sampling", total = len(self.sample_dataloader))

        effective_past_batch = None
        for idx, (past_batch, predict_batch, trend_batch, volatility_batch, liquidity_batch, imb_batch, _, predict_time_batch) in enumerate(loop):

            # AR overwrite
            if self.AR and idx > 0:
                is_reset_step = (idx % self.refresh_cycle == 0)
                if (not is_reset_step) and (effective_past_batch is not None):
                    ask_idx = torch.arange(10 - self.n_levels, 10)
                    bid_idx = torch.arange(10, 10 + self.n_levels)
                    raw_idx = torch.cat([ask_idx, bid_idx], dim = 0)
                    past_batch[:, 1:, raw_idx, :] = effective_past_batch

            transformed_past_batch, _ = transform_past_predict_batch(past_batch, predict_batch, self.n_levels)

            # responsive condition replacement
            if self.responsive_liquidity is not None:
                rand_idx = torch.randint(0, liquidity_pool.shape[0], size = liquidity_batch.shape)
                liquidity_batch = liquidity_pool[rand_idx]
            if self.responsive_imb is not None:
                rand_idx = torch.randint(0, imb_pool.shape[0], size = imb_batch.shape)
                imb_batch = imb_pool[rand_idx]
            if self.responsive_trend is not None:
                rand_idx = torch.randint(0, trend_pool.shape[0], size = trend_batch.shape)
                trend_batch = trend_pool[rand_idx]
            if self.responsive_volatility is not None:
                rand_idx = torch.randint(0, volatility_pool.shape[0], size = volatility_batch.shape)
                volatility_batch = volatility_pool[rand_idx]

            # cond construction (same scaling)
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

            # sample z ~ N(0,1), then decode
            B = past_batch.shape[0]
            z = torch.randn(B, self.z_dim, device = self.device)

            with torch.no_grad():
                samples = self.vae.decode(z, cond, F_dim = self.F, T_dim = self.predict_window)  # [B,2,F,T]

            # postprocess exactly like diffusion/gan
            samples[:, 0:1, :, :] = samples[:, 0:1, :, :] / 100
            samples[:, 0:1, :, :] = transform_sample_batch(past_batch, samples[:, 0:1, :, :], self.n_levels).permute(0, 3, 2, 1)

            samples[:, -1:, :, :] = (samples[:, -1:, :, :] * 15) ** 2

            if self.AR:
                effective_past_batch = samples.permute(0, 3, 2, 1).detach().cpu()  # [B,T,F,2]

            samples_out = samples.permute(0, 3, 2, 1)  # [B,T,F,2]
            fake_samples.append(samples_out.clone().detach())

        fake_samples = [t.cpu().numpy() for t in fake_samples]
        np.save(self.vae_samples_saving_path, fake_samples)
