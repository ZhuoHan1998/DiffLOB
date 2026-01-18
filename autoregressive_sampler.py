import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from utils.util import transform_past_predict_batch, transform_sample_batch
from nets.autoregressive_s5 import AutoregressiveS5Model


class AutoregressiveSampler:
    def __init__(self, sample_dataloader, config, **kwargs):
        self.sample_dataloader = sample_dataloader
        self.config = config

        self.n_levels = config.n_levels
        self.past_window = config.past_window
        self.predict_window = config.predict_window
        self.store_length = config.store_length
        self.sampling_batch_size = config.sampling_batch_size

        # model params
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.state_dim = config.state_dim
        self.dropout = config.dropout

        # paths
        self.model_saving_path = config.model_saving_path
        self.samples_saving_path = config.samples_saving_path

        # sampler knobs
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
        self.T = self.predict_window
        self.x_dim = 2 * self.F
        self.cond_dim = 5

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
        self.model = AutoregressiveS5Model(
            x_dim = self.x_dim,
            cond_dim = self.cond_dim,
            d_model = self.d_model,
            n_layers = self.n_layers,
            state_dim = self.state_dim,
            dropout = self.dropout,
        ).to(self.device)

        ckpt = torch.load(self.model_saving_path, map_location = self.device)
        self.model.load_state_dict(ckpt, strict = True)
        self.model.eval()

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
        loop = tqdm(self.sample_dataloader, desc = "AR-S5 Sampling", total = len(self.sample_dataloader))

        effective_past_batch = None
        for idx, (past_batch, predict_batch, trend_batch, volatility_batch, liquidity_batch, imb_batch, _, predict_time_batch) in enumerate(loop):

            # Autoregressive overwrite on past_batch
            if self.AR and idx > 0:
                is_reset_step = (idx % self.refresh_cycle == 0)
                if (not is_reset_step) and (effective_past_batch is not None):
                    ask_idx = torch.arange(10 - self.n_levels, 10)
                    bid_idx = torch.arange(10, 10 + self.n_levels)
                    raw_idx = torch.cat([ask_idx, bid_idx], dim = 0)
                    past_batch[:, 1:, raw_idx, :] = effective_past_batch

            transformed_past, _ = transform_past_predict_batch(past_batch, predict_batch, self.n_levels)

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

            # past last in normalized space -> prev_x_0
            past_price = transformed_past[:, :, :, 0:1] * 100
            past_vol = torch.sqrt(transformed_past[:, :, :, -1:]) / 15
            past_x = torch.cat([past_price, past_vol], dim = -1).permute(0, 3, 2, 1).contiguous()  # [B,2,F,Tpast]
            prev = past_x[:, :, :, -1].contiguous().view(past_x.size(0), self.x_dim).to(self.device)  # [B,x_dim]

            # cond sequences (scaled like training)
            trend = torch.where(trend_batch >= 0, torch.sqrt(trend_batch), -torch.sqrt(-trend_batch)) * 10
            vol = torch.sqrt(volatility_batch) * 10
            liq = torch.sqrt((liquidity_batch / 2)) / 15
            imb = imb_batch
            time = predict_time_batch

            cond_seq = torch.stack([
                trend.to(self.device).unsqueeze(1).expand(-1, self.T),
                vol.to(self.device).unsqueeze(1).expand(-1, self.T),
                liq.to(self.device),
                imb.to(self.device),
                time.to(self.device),
            ], dim = -1)  # [B,T,5]

            # autoregressive generation with cache
            B = past_batch.shape[0]
            caches = self.model.init_cache(B, device = torch.device(self.device))

            preds = []
            for t in range(self.T):
                cond_t = cond_seq[:, t, :]                    # [B,5]
                x_t, caches = self.model.step(prev, cond_t, caches = caches)  # [B,x_dim]
                preds.append(x_t)
                prev = x_t

            pred_seq = torch.stack(preds, dim = 1)              # [B,T,x_dim]

            # reshape back to [B,2,F,T] normalized space
            pred_x = pred_seq.view(B, self.T, 2, self.F).permute(0, 2, 3, 1).contiguous()  # [B,2,F,T]

            # postprocess to original scale (same style)
            pred_x[:, 0:1, :, :] = pred_x[:, 0:1, :, :] / 100
            pred_x[:, 0:1, :, :] = transform_sample_batch(past_batch, pred_x[:, 0:1, :, :], self.n_levels).permute(0, 3, 2, 1)

            pred_x[:, -1:, :, :] = (pred_x[:, -1:, :, :] * 15) ** 2

            if self.AR:
                # store for overwrite: [B,T,F,2]
                effective_past_batch = pred_x.permute(0, 3, 2, 1).detach().cpu()

            samples_out = pred_x.permute(0, 3, 2, 1)  # [B,T,F,2]
            fake_samples.append(samples_out.detach().cpu().numpy())

        np.save(self.samples_saving_path, fake_samples)
