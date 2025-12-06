import pandas as pd
import numpy as np
import random
import torch
from tqdm import tqdm

from utils.util import transform_sample_batch, inverse_nse_transform
from utils.sde_lib import VESDE, VPSDE, subVPSDE
from utils.sampler_func import get_sampling_fn


class Sampler():
    """
    In every sampling, we can get snapshots of length = predict_window.
    """
    def __init__(self, sample_dataloader, config, **kwargs,):
        
        self.sample_dataloader = sample_dataloader
        self.config = config
        self.sampling_batch_size = config.sampling_batch_size
        self.diff_model = config.diff_model
        self.diff_model_loading_path = config.diff_model_saving_path
        self.samples_saving_path = config.samples_saving_path
        self.guidance = config.guidance
        self.past_window = config.past_window
        self.predict_window = config.predict_window
        self.store_length = config.store_length 
        self.md_type = config.md_type
        self.AR = config.AR
        self.responsive_liquidity = config.responsive_liquidity
        self.responsive_trend = config.responsive_trend
        self.responsive_volatility = config.responsive_volatility
        self.liquidity_cond_path = config.liquidity_cond_path
        self.trend_cond_path = config.trend_cond_path
        self.volatility_cond_path = config.volatility_cond_path
        self.bin_mode = config.bin_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # determine diffusion process
        if self.md_type == 'vesde':
            self.sde = VESDE(sigma_min = config.sigma_min, sigma_max = config.sigma_max, N = config.num_scales)
            self.sampling_eps = 1e-5
        elif self.md_type == 'vpsde':
            self.sde = VPSDE(beta_min = config.beta_min, beta_max = config.beta_max, N = config.num_scales)
            self.sampling_eps = 1e-3
        elif self.md_type == 'subvpsde':
            self.sde = subVPSDE(beta_min = config.beta_min, beta_max = config.beta_max, N = config.num_scales)
            self.sampling_eps = 1e-3
        
        self.shape = (self.sampling_batch_size, 2, 20, self.predict_window)  # shape of score network input
            
    def get_conditional_pool(self, npy_path, selector, mode = "quantile", q = 5):
        """Select responsive conditions by either quantile or range
           If selector = 'low', choose lowest bin; else choose highest bin 
        """
        responsive_cond = np.load(npy_path)
        if mode == "quantile":
            bins = pd.qcut(responsive_cond, q = q, labels = False)
        elif mode == "range":
            bins = pd.cut(responsive_cond, bins = q, labels = False)
        target = 0 if selector == "low" else q - 1
        pool = responsive_cond[bins == target]

        return torch.tensor(pool, dtype = torch.float32)

    def sample(self):
        
        sampling_fn = get_sampling_fn(self.config, self.sde, self.shape, self.sampling_eps, self.device)
        
        # load model
        if self.diff_model == "csdi":
            from nets.diff_csdi import diff_CSDI
            self.diff_net = diff_CSDI(inputdim = 2, side_dim = 9)
        if self.diff_model == "s4":
            from nets.diff_s4 import diff_S4
            self.diff_net = diff_S4(input_dim = 2, cond_dim = 9)
        if self.diff_model == "transformer":
            from nets.transformer import TransformerDiffusionModel
            self.diff_net = TransformerDiffusionModel()
        if self.diff_model == "unet":
            from nets.unet import CNNDiffusionModel
            self.diff_net = CNNDiffusionModel()
        if self.diff_model == "wavenet":
            from nets.diff_wavenet_joint import WaveNetJoint
            self.diff_net = WaveNetJoint(input_dim = 2, cond_dim = 9)
            
        diff_ckpt = torch.load(self.diff_model_loading_path, map_location = self.device, weights_only = True)
        self.diff_net.load_state_dict(diff_ckpt, strict = True, assign = True)
        self.diff_net.to(self.device)
        self.diff_net.eval()
        
        # build responsive liquidity conditions
        if self.responsive_liquidity is not None:
            selector = self.responsive_liquidity
            liquidity_pool = self.get_conditional_pool(self.liquidity_cond_path, self.responsive_liquidity, mode = self.bin_mode)

        # build responsive trend conditions
        if self.responsive_trend is not None:
            selector = self.responsive_trend
            trend_pool = self.get_conditional_pool(self.trend_cond_path, self.responsive_trend, mode = self.bin_mode)

        # build responsive volatility conditions
        if self.responsive_volatility is not None:
            selector = self.responsive_volatility
            volatility_pool = self.get_conditional_pool(self.volatility_cond_path, self.responsive_volatility, mode = self.bin_mode)
        
        # iterative sampling with tqdm progress bar
        fake_samples = []
        loop = tqdm(self.sample_dataloader, desc = "Sampling", total = len(self.sample_dataloader))
        
        # use effective_past_batch as buffer
        effective_past_batch = None
        for idx, (past_batch, _, volatility_batch, trend_batch, liquidity_batch, oi_batch, _, predict_time_batch) in enumerate(loop):
            
            if self.AR:
                if idx > 0:
                    # we only take autoregressive sampling on price, not on volume
                    past_batch[:, :, :, 0] = effective_past_batch[:, :, :, 0]
                    
            # select random responsive liquidity conditions
            if self.responsive_liquidity is not None:
                rand_idx = torch.randint(0, liquidity_pool.shape[0], size = liquidity_batch.shape)
                liquidity_batch = liquidity_pool[rand_idx]

            # select random responsive trend conditions
            if self.responsive_trend is not None:
                rand_idx = torch.randint(0, trend_pool.shape[0], size = trend_batch.shape)
                trend_batch = trend_pool[rand_idx]

            # select random responsive volatility conditions
            if self.responsive_volatility is not None:
                rand_idx = torch.randint(0, volatility_pool.shape[0], size = volatility_batch.shape)
                volatility_batch = volatility_pool[rand_idx]
            
            # Select condition
            # Build each condition tensor of shape [B, 1, 20, predict_window]

            past_price_cond = past_batch[:, :, :, 0].permute(0, 2, 1).unsqueeze(1)
            past_price_cond = past_price_cond / 100

            past_volume_cond = past_batch[:, :, :, 1].permute(0, 2, 1).unsqueeze(1)
            past_volume_cond = torch.sqrt(past_volume_cond) / 15 
            
            volatility_cond = volatility_batch.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, self.predict_window, 20).permute(0, 1, 3, 2)
            # volatility_cond = volatility_cond * 1e1
            
            trend_cond = trend_batch.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, self.predict_window, 20).permute(0, 1, 3, 2)
            # trend_cond = trend_cond * 1e1

            liquidity_cond = liquidity_batch / 20
            liquidity_cond = liquidity_cond.unsqueeze(1).unsqueeze(-1).expand(-1, 1, self.config.predict_window, 20).permute(0, 1, 3, 2)
            liquidity_cond = torch.sqrt(liquidity_cond) / 15
            
            oi_cond = oi_batch.unsqueeze(1).unsqueeze(-1).expand(-1, 1, self.config.predict_window, 20).permute(0, 1, 3, 2)

            time_cond = predict_time_batch.unsqueeze(1).unsqueeze(-1).expand(-1, 1, self.config.predict_window, 20).permute(0, 1, 3, 2)

            # add past mid price info into condition
            past_mid_cond = (past_price_cond[:, :, 9, :] + past_price_cond[:, :, 10, :]) / 2 # [B, 1, past_window]
            past_mid_cond = past_mid_cond.unsqueeze(2).expand(-1, -1, 20, -1) # [B, 1, 20, past_window]

            # add past mid price difference info into condition
            past_mid_diff_cond = torch.diff(past_mid_cond, dim = -1, prepend = past_mid_cond[..., :1]) # [B, 1, 20, past_window]
            # past_mid_diff_cond = past_mid_diff_cond * 1e2

            assert (past_price_cond.shape[1] == past_volume_cond.shape[1] == volatility_cond.shape[1] == trend_cond.shape[1] == liquidity_cond.shape[1] == oi_cond.shape[1] == time_cond.shape[1] == past_mid_cond.shape[1] == past_mid_diff_cond.shape[1]), "Past condition length does not match prediction length."
            cond = torch.cat([past_price_cond, past_volume_cond, volatility_cond, trend_cond, liquidity_cond, oi_cond, time_cond, past_mid_cond, past_mid_diff_cond], dim = 1)
            
            cond = cond.to(self.device)
            
            # single sampling
            # sampling_fn returns two tensors: x_mean and sde.N * (n_steps + 1)
            samples = sampling_fn(self.diff_net, cond, self.guidance)[0]   # shape: [batch_size, 2, 20, predict_window]
            
            # Norm-salvaged Embedding for return series
#             samples[:, 0, 0, 0] = samples[:, 0, 0, 0]
#             samples[:, 0, 0, 1] = samples[:, 0, 0, 1]
#             samples[:, 0, 1:, :] = samples[:, 0, 1:, :]
            
#             for i in range(samples.shape[0]):
#                 embedded_series = samples[i, 0, 0, :]
#                 transformed_series = inverse_nse_transform(embedded_series)
#                 samples[i, 0, 0, :] = transformed_series
            
            
            # price postprocess
            samples[:, 0, :, :] = transform_sample_batch(past_batch, samples[:, 0, :, :].unsqueeze(1))
            # samples[:, 0, :, :] = samples[:, 0, :, :] / 1e2

            # volume process
            samples[:, 1, :, :] = (samples[:, 1, :, :].unsqueeze(1) * 15) ** 2
                
            # generated samples should be past_batch in the next iteration
            if self.AR:
                effective_past_batch = samples.permute(0, 3, 2, 1).detach().cpu()
            
            # reshape samples to [batch_size, predict_window, 20, 2]
            samples = samples.permute(0, 3, 2, 1)
            
            # Store a copy of samples in fake_samples list.
            fake_samples.append(samples.clone().detach())
            
        # store fake_samples as npy file
        fake_samples = [tensor.cpu().numpy() for tensor in fake_samples]
        np.save(self.samples_saving_path, fake_samples)
    
    
