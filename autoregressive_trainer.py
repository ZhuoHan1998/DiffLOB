import torch
import torch.nn as nn
from torch.optim import Adam
import tqdm

from utils.util import transform_past_predict_batch
from nets.autoregressive_s5 import AutoregressiveS5Model


class AutoregressiveTrainer:
    def __init__(self, train_dataloader, val_dataloader, config, **kwargs):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        self.n_levels = config.n_levels
        self.past_window = config.past_window
        self.predict_window = config.predict_window

        # model params
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.state_dim = config.state_dim
        self.dropout = config.dropout

        # training params
        self.n_epochs = config.n_epochs
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.clip_gradient = config.clip_gradient
        self.recon_loss_type = config.recon_loss_type

        self.early_stop_patience = config.early_stop_patience
        self.early_stop_min_delta = config.early_stop_min_delta
        self.model_saving_path = config.model_saving_path

        self.drop_probability = config.drop_probability

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.F = 2 * self.n_levels
        self.x_dim = 2 * self.F          # flatten 2 x F
        self.cond_dim = 5                # trend, vol, liq_t, imb_t, time_t

        if self.recon_loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.L1Loss()

    def _process_batch(self, past_batch, predict_batch, trend_batch, volatility_batch, liquidity_batch, imb_batch, past_time_batch, predict_time_batch):
        """
        Match your diffusion scaling and return:
          x_future_seq: [B,T,x_dim]
          x_prev_seq:   [B,T,x_dim] (teacher forcing inputs)
          cond_seq:     [B,T,cond_dim]
          plus raw past_batch for postprocess compatibility if needed.
        """
        transformed_past, transformed_pred = transform_past_predict_batch(
            past_batch, predict_batch, self.n_levels
        )
        
        transformed_past = transformed_past.to(self.device)
        transformed_pred = transformed_pred.to(self.device)

        # --- build x_future in normalized space, shape [B,2,F,T] ---
        x_price = transformed_pred[:, :, :, 0:1] * 100
        x_vol = torch.sqrt(transformed_pred[:, :, :, -1:]) / 15
        x = torch.cat([x_price, x_vol], dim = -1).permute(0, 3, 2, 1).contiguous()  # [B,2,F,T]
        x = x.to(self.device)

        # flatten to [B,T,2F]
        x_seq = x.permute(0, 3, 1, 2).contiguous().view(x.size(0), self.predict_window, self.x_dim)  # [B,T,x_dim]

        # past last step in normalized space (for prev_x[0])
        past_price = transformed_past[:, :, :, 0:1] * 100
        past_vol = torch.sqrt(transformed_past[:, :, :, -1:]) / 15
        past_x = torch.cat([past_price, past_vol], dim=-1).permute(0, 3, 2, 1).contiguous()  # [B,2,F,Tpast]
        past_last = past_x[:, :, :, -1]  # [B,2,F]
        past_last = past_last.contiguous().view(past_last.size(0), self.x_dim)  # [B,x_dim]

        # teacher forcing prev_x: [past_last, x_seq[:-1]]
        prev_x = torch.cat([past_last.unsqueeze(1), x_seq[:, :-1, :]], dim=1)  # [B,T,x_dim]

        # cond scaling (same as diffusion/gan)
        trend = torch.where(trend_batch >= 0, torch.sqrt(trend_batch), -torch.sqrt(-trend_batch)) * 10
        vol = torch.sqrt(volatility_batch) * 10
        liq = torch.sqrt((liquidity_batch / 2)) / 15
        imb = imb_batch
        time = predict_time_batch

        # cond_seq: [B,T,5]
        cond_seq = torch.stack([
            trend.to(self.device).unsqueeze(1).expand(-1, self.predict_window),
            vol.to(self.device).unsqueeze(1).expand(-1, self.predict_window),
            liq.to(self.device),
            imb.to(self.device),
            time.to(self.device),
        ], dim = -1)

        # optional cond dropout
        if self.drop_probability > 0 and torch.rand(1).item() < self.drop_probability:
            cond_seq = torch.zeros_like(cond_seq)
            
        x_seq = torch.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0)
        prev_x = torch.nan_to_num(prev_x, nan=0.0, posinf=0.0, neginf=0.0)
        cond_seq = torch.nan_to_num(cond_seq, nan=0.0, posinf=0.0, neginf=0.0)

        return x_seq, prev_x, cond_seq

    def train(self):
        self.model = AutoregressiveS5Model(
            x_dim = self.x_dim,
            cond_dim = self.cond_dim,
            d_model = self.d_model,
            n_layers = self.n_layers,
            state_dim = self.state_dim,
            dropout = self.dropout,
        ).to(self.device)

        opt = Adam(self.model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        scaler = torch.amp.GradScaler()

        best_val = float("inf")
        early_stop_counter = 0

        loop = tqdm.trange(self.n_epochs)
        for epoch in loop:
            # ---------------- train ----------------
            self.model.train()
            train_loss = 0.0
            n_items = 0

            for batch in self.train_dataloader:
                (past_batch, predict_batch, trend_batch, volatility_batch,
                 liquidity_batch, imb_batch, past_time_batch, predict_time_batch) = batch

                x_seq, prev_x, cond_seq = self._process_batch(
                    past_batch, predict_batch, trend_batch, volatility_batch,
                    liquidity_batch, imb_batch, past_time_batch, predict_time_batch
                )
                B = x_seq.size(0)
                n_items += B

                opt.zero_grad(set_to_none = True)

                pred = self.model(prev_x, cond_seq)          # [B, T, x_dim]
                loss = self.loss_fn(pred, x_seq)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                if self.clip_gradient is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)
                opt.step()

                train_loss += loss.item() * B

            train_loss = train_loss / max(n_items, 1)

            # ---------------- val ----------------
            self.model.eval()
            val_loss = 0.0
            val_items = 0
            with torch.no_grad():
                for batch in self.val_dataloader:
                    (past_batch, predict_batch, trend_batch, volatility_batch,
                     liquidity_batch, imb_batch, past_time_batch, predict_time_batch) = batch

                    x_seq, prev_x, cond_seq = self._process_batch(
                        past_batch, predict_batch, trend_batch, volatility_batch,
                        liquidity_batch, imb_batch, past_time_batch, predict_time_batch
                    )
                    B = x_seq.size(0)
                    val_items += B

                    pred = self.model(prev_x, cond_seq)
                    loss = self.loss_fn(pred, x_seq)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    val_loss += loss.item() * B

            val_loss = val_loss / max(val_items, 1)

            loop.set_description(f"Epoch {epoch} | Train={train_loss:.6f} | Val={val_loss:.6f}")

            # save / early stop
            if best_val - val_loss > self.early_stop_min_delta:
                best_val = val_loss
                early_stop_counter = 0
                torch.save(self.model.state_dict(), self.model_saving_path)
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch}, best val={best_val}")
                break
