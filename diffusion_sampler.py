import pandas as pd
import numpy as np
import random
import torch
from tqdm import tqdm

from utils.util import transform_past_predict_batch, transform_sample_batch
from utils.sde_lib import VESDE, VPSDE, subVPSDE
from utils.sampler_func import get_sampling_fn

class Sampler():
    """
    In every sampling, we can get snapshots of length = predict_window.
    """
    def __init__(self, sample_dataloader, config, **kwargs,):
        
        self.sample_dataloader = sample_dataloader
        self.config = config
        self.n_levels = config.n_levels
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
        self.control = config.control
        self.motion = config.motion
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
        
        self.F = 2 * self.n_levels
        self.shape = (self.sampling_batch_size, 2, self.F, self.predict_window)  # shape of score network input
            
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
            self.diff_net = diff_CSDI(inputdim = 2, side_dim = 7)
        if self.diff_model == "s4":
            from nets.diff_s4 import diff_S4
            self.diff_net = diff_S4(input_dim = 2, cond_dim = 7)
        if self.diff_model == "wavenet":
            from nets.diff_wavenet import WaveNetJoint
            self.diff_net = WaveNetJoint(input_dim = 2)
        if self.diff_model == "wavenet_motion":
            from nets.diff_wavenet_motion import WaveNetJoint
            self.diff_net = WaveNetJoint(input_dim = 2)
        if self.diff_model == "wavenet_control":
            from nets.diff_wavenet_control import WaveNetJoint
            self.diff_net = WaveNetJoint(input_dim = 2)
        if self.diff_model == "wavenet_motion_control":
            from nets.diff_wavenet_motion_control import WaveNetJoint
            self.diff_net = WaveNetJoint(input_dim = 2)
            
        diff_ckpt = torch.load(self.diff_model_loading_path, map_location = self.device, weights_only = True)
        self.diff_net.load_state_dict(diff_ckpt, strict = True, assign = True)
        self.diff_net.to(self.device)
        self.diff_net.eval()
        
        # build responsive liquidity conditions
        if self.responsive_liquidity is not None:
            liquidity_pool = self.get_conditional_pool(self.liquidity_cond_path, self.responsive_liquidity, self.bin_mode)
            
        # build responsive imbalance conditions
        if self.responsive_imb is not None:
            imb_pool = self.get_conditional_pool(self.imb_cond_path, self.responsive_imb, self.bin_mode)

        # build responsive trend conditions
        if self.responsive_trend is not None:
            trend_pool = self.get_conditional_pool(self.trend_cond_path, self.responsive_trend, self.bin_mode)

        # build responsive volatility conditions
        if self.responsive_volatility is not None:
            volatility_pool = self.get_conditional_pool(self.volatility_cond_path, self.responsive_volatility, self.bin_mode)
        
        # iterative sampling with tqdm progress bar
        fake_samples = []
        loop = tqdm(self.sample_dataloader, desc = "Sampling", total = len(self.sample_dataloader))
        
        # use effective_past_batch as buffer for generated batch
        effective_past_batch = None
        for idx, (past_batch, predict_batch, trend_batch, volatility_batch, liquidity_batch, imb_batch, _, predict_time_batch) in enumerate(loop):
            
            # For every refresh_cycle sampling, model uses real condition, otherelse takes generated data as conditions.
            refresh_cycle = self.refresh_cycle
            if self.AR and idx > 0:
                is_reset_step = (idx % refresh_cycle == 0)

                # only overwrite when not reset and we already have generated buffer
                if (not is_reset_step) and (effective_past_batch is not None):

                    # raw 20-level indices for the selected levels
                    ask_idx = torch.arange(10 - self.n_levels, 10)          # [n_levels]
                    bid_idx = torch.arange(10, 10 + self.n_levels)          # [n_levels]
                    raw_idx = torch.cat([ask_idx, bid_idx], dim = 0)        # [F]

                    # replace selected levels at each step
                    past_batch[:, 1:, raw_idx, :] = effective_past_batch
            
            # both shape is [B, predict_window, 20, 2] if 'past_window - 1 == predict_window'
            transformed_past_batch, _ = transform_past_predict_batch(past_batch, predict_batch, self.n_levels) # is there data leakage here? 
            
            # select random responsive liquidity conditions
            if self.responsive_liquidity is not None:
                rand_idx = torch.randint(0, liquidity_pool.shape[0], size = liquidity_batch.shape)
                liquidity_batch = liquidity_pool[rand_idx]
                
            # select random responsive imbalance conditions
            if self.responsive_imb is not None:
                rand_idx = torch.randint(0, imb_pool.shape[0], size = imb_batch.shape)
                imb_batch = imb_pool[rand_idx]

            # select random responsive trend conditions
            if self.responsive_trend is not None:
                rand_idx = torch.randint(0, trend_pool.shape[0], size = trend_batch.shape)
                trend_batch = trend_pool[rand_idx]

            # select random responsive volatility conditions
            if self.responsive_volatility is not None:
                rand_idx = torch.randint(0, volatility_pool.shape[0], size = volatility_batch.shape)
                volatility_batch = volatility_pool[rand_idx]
            
            # ---------- conditions cond : [B, 7, F, predict_window] ----------
            
            past_price_cond = transformed_past_batch[:, :, :, 0:1].permute(0, 3, 2, 1)
            past_price_cond = past_price_cond * 100

            past_volume_cond = transformed_past_batch[:, :, :, -1:].permute(0, 3, 2, 1)
            past_volume_cond = torch.sqrt(past_volume_cond) / 15
            
            trend_cond = trend_batch
            trend_cond = torch.where(trend_cond >= 0, torch.sqrt(trend_cond), -torch.sqrt(-trend_cond)) * 10
            
            volatility_cond = volatility_batch
            volatility_cond = torch.sqrt(volatility_cond) * 10

            liquidity_cond = liquidity_batch / 2
            liquidity_cond = liquidity_cond
            liquidity_cond = torch.sqrt(liquidity_cond) / 15
            
            imb_cond = imb_batch

            time_cond = predict_time_batch
            
            cond = (past_price_cond.to(self.device), past_volume_cond.to(self.device), 
                    trend_cond.to(self.device), volatility_cond.to(self.device), 
                    liquidity_cond.to(self.device), imb_cond.to(self.device), 
                    time_cond.to(self.device))
            
            # single sampling
            # sampling_fn returns two tensors: x_mean and sde.N * (n_steps + 1)
            if self.diff_model in ["wavenet", "csdi", "s4"]:
                samples = sampling_fn(self.diff_net, cond, self.guidance, enable_motion = False, enable_control = False)[0]   # [B, 2, F, T]
                
            else:
                samples = sampling_fn(self.diff_net, cond, self.guidance, enable_motion = self.motion, enable_control = self.control)[0]   # [B, 2, F, T]
            
            # price postprocess
            samples[:, 0:1, :, :] = samples[:, 0:1, :, :] / 100
            samples[:, 0:1, :, :] = transform_sample_batch(past_batch, samples[:, 0:1, :, :], self.n_levels).permute(0, 3, 2, 1)
            
            # volume postprocess
            samples[:, -1:, :, :] = (samples[:, -1:, :, :] * 15) ** 2
                
            # generated samples should be past_batch in the next iteration
            if self.AR:
                effective_past_batch = samples.permute(0, 3, 2, 1).detach().cpu()  # [B, T, F, 2]
            
            # reshape samples to [B, T, F, 2]
            samples = samples.permute(0, 3, 2, 1)
            
            # Store a copy of samples in fake_samples list.
            fake_samples.append(samples.clone().detach())
            
        # store fake_samples as npy file
        fake_samples = [tensor.cpu().numpy() for tensor in fake_samples]
        np.save(self.samples_saving_path, fake_samples)
    
    
