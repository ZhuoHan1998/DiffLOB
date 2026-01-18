import torch
import torch.nn as nn
from torch.optim import Adam
import tqdm

from utils.util import transform_past_predict_batch, nse_transform
from utils.sde_lib import VESDE, VPSDE, subVPSDE
from utils.losses import get_loss_fn


class Trainer():
    def __init__(self, train_dataloader, val_dataloader, config, **kwargs,):
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.n_levels = config.n_levels
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
        self.control = config.control
        self.motion = config.motion
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
        
        
    def _process_batch(self, past_batch, predict_batch, trend_batch, volatility_batch, liquidity_batch, imb_batch, past_time_batch, predict_time_batch):
        
        # both shape is [B, predict_window, F, 2] if 'past_window - 1 == predict_window'
        transformed_past_batch, transformed_predict_batch = transform_past_predict_batch(past_batch, predict_batch, self.n_levels) 
        _, _, F, _ = transformed_past_batch.shape
        
        # ---------- target x : [B, 2, F, predict_window] ----------

        # price process
        x_price = transformed_predict_batch[:, :, :, 0:1] # [B, predict_window, F, 1]
        x_price = x_price * 100
        
        # volume process
        x_volume = transformed_predict_batch[:, :, :, -1:]
        x_volume = torch.sqrt(x_volume) / 15 # [B, predict_window, F, 1]
        
        # concat price and volume
        x = torch.concat([x_price, x_volume], dim = -1).permute(0, 3, 2, 1)       
        x = x.to(self.device)
        
        # ---------- conditions cond : [B, 7, F, predict_window] ----------
        
        past_price_cond = transformed_past_batch[:, :, :, 0:1].permute(0, 3, 2, 1)
        past_price_cond = past_price_cond * 100

        past_volume_cond = transformed_past_batch[:, :, :, -1:].permute(0, 3, 2, 1)
        past_volume_cond = torch.sqrt(past_volume_cond) / 15
        
        trend_cond = trend_batch
        # trend_cond = torch.clamp(trend_cond, min = -0.1, max = 0.1)
        trend_cond = torch.where(trend_cond >= 0, torch.sqrt(trend_cond), -torch.sqrt(-trend_cond)) * 10 # what if we use multiplication of 100?
        
        volatility_cond = volatility_batch
        # volatility_cond = torch.clamp(volatility_cond, max = 0.1)
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
        
        # Classifier-free guidance dropout
        if torch.rand(1).item() < self.drop_probability:
            cond = tuple(torch.zeros_like(item) for item in cond)
        
        return x, cond
    

    def _run_epoch(self, dataloader, training: bool, enable_motion: bool, enable_control:bool):
        net = self.diff_net
        if training:
            net.train()      
        else:
            net.eval()

        total_loss = 0.0
        total_items = 0
        scaler = torch.amp.GradScaler()

        for past_batch, predict_batch, trend_batch, volatility_batch, liquidity_batch, imb_batch, past_time_batch, predict_time_batch in dataloader:
            x, cond = self._process_batch(past_batch, predict_batch, trend_batch, volatility_batch, liquidity_batch, imb_batch, past_time_batch, predict_time_batch)

            if training:    
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type = self.device):
                    loss = self.loss_fn(net, x, cond, enable_motion, enable_control)
                    
                # skip batch that would elicit nan loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad()
                    continue
                    
                scaler.scale(loss).backward()
                
                if self.clip_gradient is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)
                                    
                scaler.step(self.optimizer)
                scaler.update()
            
            else:
                with torch.no_grad():
                    loss = self.loss_fn(net, x, cond, enable_motion, enable_control)

            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_items += batch_size

        return total_loss / total_items if total_items > 0 else float('inf')

    def train(self):
        
        # Initialize model
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
        
        self.diff_net.to(self.device)
        net = self.diff_net
                    
        def each_layer_train_val(enable_motion, enable_control):
            for param in net.parameters():
                param.requires_grad = True

            if enable_control == True and enable_motion == False:
                # When solely training control: freeze all except control blocks
                for name, param in net.named_parameters():
                    if "control_blocks" not in name:
                        param.requires_grad = False
            
            if enable_control == False and enable_motion == True:
                # When solely training motion: freeze all except control blocks
                for name, param in net.named_parameters():
                    if "motion_blocks" not in name:
                        param.requires_grad = False
                        
            if enable_control == True and enable_motion == True:
                # When training control and motion: freeze all except control and motion blocks
                for name, param in net.named_parameters():
                    if "control_blocks" not in name and "motion_blocks" not in name:
                        param.requires_grad = False
        
            best_loss = float('inf')
            early_stop_counter = 0
            loop = tqdm.trange(self.n_epochs)
            self.optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), 
                                  lr = self.learning_rate, weight_decay = self.weight_decay) 

            for epoch in loop:
                train_loss = self._run_epoch(self.train_dataloader, training = True, enable_motion = enable_motion, enable_control = enable_control)
                val_loss = self._run_epoch(self.val_dataloader, training = False, enable_motion = enable_motion, enable_control = enable_control)

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
                    print(f"Early stopping triggered at epoch {epoch}, best loss is {best_loss}.")
                    break
                    
        # train spatial layer
        print("Starting training spatial layer")
        each_layer_train_val(enable_motion = False, enable_control = False)
        
        if self.control == True and self.motion == False:
            print("Starting training control layer")
            # load best model from last training phase
            ckpt = torch.load(self.diff_model_saving_path, map_location = self.device)
            self.diff_net.load_state_dict(ckpt)
            each_layer_train_val(enable_motion = False, enable_control = True)
            
        if self.control == False and self.motion == True:
            print("Starting training motion layer")
            # load best model from last training phase
            ckpt = torch.load(self.diff_model_saving_path, map_location = self.device)
            self.diff_net.load_state_dict(ckpt)
            each_layer_train_val(enable_motion = True, enable_control = False)        
        
        if self.control == True and self.motion == True:
            print("Starting training motion layer")
            # load best model from last training phase
            ckpt = torch.load(self.diff_model_saving_path, map_location = self.device)
            self.diff_net.load_state_dict(ckpt)
            each_layer_train_val(enable_motion = True, enable_control = False)
            
            print("Starting training control layer")
            # load best model from last training phase
            ckpt = torch.load(self.diff_model_saving_path, map_location = self.device)
            self.diff_net.load_state_dict(ckpt)
            each_layer_train_val(enable_motion = False, enable_control = True)
            

            
            