import torch
import torch.nn as nn
from torch.optim import Adam
import tqdm

from utils.util import ExponentialMovingAverage, transform_predict_batch, nse_transform
from utils.sde_lib import VESDE, VPSDE, subVPSDE
from utils.losses import get_loss_fn


class Trainer():
    def __init__(self, train_dataloader, val_dataloader, config, **kwargs,):
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.n_epochs = config.n_epochs
        self.learning_rate = config.learning_rate
        self.diff_model = config.diff_model
        self.diff_model_saving_path = config.diff_model_saving_path
        self.past_window = config.past_window
        self.predict_window = config.predict_window
        self.drop_probability = config.drop_probability
        self.weight_decay = config.weight_decay
        self.clip_gradient = config.clip_gradient
        self.ema_rate = config.ema_rate
        self.md_type = config.md_type
        self.continuous = config.continuous
        self.reduce_mean = config.reduce_mean
        self.likelihood_weighting = config.likelihood_weighting
        self.early_stop_patience = config.early_stop_patience
        self.early_stop_min_delta = config.early_stop_min_delta
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # determine diffusion process
        if self.md_type == 'vesde':
            self.sde = VESDE(sigma_min = config.sigma_min, sigma_max = config.sigma_max, N = config.num_scales)
        elif self.md_type == 'vpsde':
            self.sde = VPSDE(beta_min = config.beta_min, beta_max = config.beta_max, N = config.num_scales)
        elif self.md_type == 'subvpsde':
            self.sde = subVPSDE(beta_min = config.beta_min, beta_max = config.beta_max, N = config.num_scales)
        
        # obtain loss function dependent on sde form. These three hyperparameters also affects loss function.
        self.loss_fn = get_loss_fn(self.sde,
                                   continuous = self.continuous, 
                                   reduce_mean = self.reduce_mean, 
                                   likelihood_weighting = self.likelihood_weighting)
        
        
    def _process_batch(self, past_batch, predict_batch, volatility_batch, trend_batch, liquidity_batch, oi_batch, past_time_batch, predict_time_batch):
        
        transformed_predict_batch = transform_predict_batch(past_batch, predict_batch) # [B, predict_window, 20, 2]

        # Build target tensor x of shape [B, 2, 20, predict_window]
        
        # price process
        x_price = transformed_predict_batch[:, :, :, 0] # [B, predict_window, 20]
        # x_price = x_price * 1e2
        
        # Norm-salvaged Embedding for return series
#         for i in range(x_price.shape[0]):
#             original_series = x_price[i, :, 0]
#             transformed_series = nse_transform(original_series)
#             x_price[i, :, 0] = transformed_series
            
#         x_price[:, 0, 0] = x_price[:, 0, 0]
#         x_price[:, 1, 0] = x_price[:, 1, 0]
#         x_price[:, :, 1:] = x_price[:, :, 1:]
        
        # volume process
        x_volume = predict_batch[:, :, :, 1]
        x_volume = torch.sqrt(x_volume) / 15
        
        x_price = x_price.unsqueeze(1).permute(0, 1, 3, 2)
        x_volume = x_volume.unsqueeze(1).permute(0, 1, 3, 2)
        
        # concat price and volume
        x = torch.cat([x_price, x_volume], dim = 1)       
        x = x.to(self.device)
        
        # Select condition
        # Build each condition tensor of shape [B, 1, 20, predict_window]
        
        past_price_cond = past_batch[:, :, :, 0].permute(0, 2, 1).unsqueeze(1) 
        past_price_cond = past_price_cond / 100

        past_volume_cond = past_batch[:, :, :, 1].permute(0, 2, 1).unsqueeze(1)
        past_volume_cond = torch.sqrt(past_volume_cond) / 15
        
        volatility_cond = volatility_batch.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, self.predict_window, 20).permute(0, 1, 3, 2)
        # volatility_cond = torch.clamp(volatility_cond, max = 0.01)
        # volatility_cond = volatility_cond * 1e1
        
        trend_cond = trend_batch.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, self.predict_window, 20).permute(0, 1, 3, 2)
        # trend_cond = torch.clamp(trend_cond, min = -0.1, max = 0.1)
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

        # concat all contions
        assert (past_price_cond.shape[1] == past_volume_cond.shape[1]  == volatility_cond.shape[1] == trend_cond.shape[1] == liquidity_cond.shape[1] == oi_cond.shape[1] == time_cond.shape[1] == past_mid_cond.shape[1] == past_mid_diff_cond.shape[1]), "Past condition length does not match prediction length."
        cond = torch.cat([past_price_cond, past_volume_cond, volatility_cond, trend_cond, liquidity_cond, oi_cond, time_cond, past_mid_cond, past_mid_diff_cond],  dim = 1)

        cond = cond.to(self.device)
        
        # Classifier-free guidance dropout
        if torch.rand(1).item() < self.drop_probability:
            cond = torch.zeros_like(cond)
        
        # x.shape = [batch_size, 2, 20, predict_window]
        # cond.shape = [batch_size, 9, 20, predict_window]
        return x, cond
    

    def _run_epoch(self, dataloader, training: bool):
        net = self.diff_net
        if training:
            net.train()
        else:
            net.eval()

        total_loss = 0.0
        total_items = 0

        for past_batch, predict_batch, volatility_batch, trend_batch, liquidity_batch, oi_batch, past_time_batch, predict_time_batch in dataloader:
            x, cond = self._process_batch(past_batch, predict_batch, volatility_batch, trend_batch, liquidity_batch, oi_batch, past_time_batch, predict_time_batch)

            if training:
                self.optimizer.zero_grad()
                loss = self.loss_fn(net, x, cond)
                # skip batch that would elicit nan loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                if self.clip_gradient is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)
                self.optimizer.step()
                if self.ema_rate is not None:
                    self.diff_ema.update(net.parameters())
            else:
                with torch.no_grad():
                    loss = self.loss_fn(net, x, cond)

            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_items += batch_size

        return total_loss / total_items if total_items > 0 else float('inf')

    def train(self):
        
        # Initialize model
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

        self.diff_net.to(self.device)

        # EMA & optimizer
        self.diff_ema = ExponentialMovingAverage(self.diff_net.parameters(), self.ema_rate)
        self.optimizer = Adam(self.diff_net.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)

        best_loss = float('inf')
        early_stop_counter = 0
        loop = tqdm.trange(self.n_epochs)

        for epoch in loop:
            train_loss = self._run_epoch(self.train_dataloader, training = True)
            val_loss = self._run_epoch(self.val_dataloader, training = False)

            loop.set_description(f"Epoch {epoch} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

            # Early stopping based on validation loss
            if best_loss - val_loss > self.early_stop_min_delta:
                best_loss = val_loss
                early_stop_counter = 0
                # save the model only when val loss decreases
                torch.save(self.diff_net.state_dict(), self.diff_model_saving_path)
                
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break


            
            
            
            