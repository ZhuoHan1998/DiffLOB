import numpy as np
import importlib.util

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data  

from .sde_lib import VESDE, VPSDE, subVPSDE

def load_config(path):
    """Load a config file as a module."""
    spec = importlib.util.spec_from_file_location("user_config", path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module

def convert_dataframe_to_tensor(df):
    """
    Convert LOB DataFrame into tensor of shape [N, 20, 3]

    Args:
        df: pd.DataFrame with columns like:
            Time_dt, Ask_Price_1~10, Ask_Size_1~10, Bid_Price_1~10, Bid_Size_1~10, Time

    Returns:
        torch.Tensor of shape [N, 20, 3]
    """
    N = len(df)

    # Extract and sort columns
    ask_price_cols = [f"Ask_Price_{i}" for i in range(10, 0, -1)]
    ask_size_cols  = [f"Ask_Size_{i}" for i in range(10, 0, -1)]
    bid_price_cols = [f"Bid_Price_{i}" for i in range(1, 11)]
    bid_size_cols  = [f"Bid_Size_{i}" for i in range(1, 11)]

    # Extract data
    ask_prices = df[ask_price_cols].values   # [N, 10]
    ask_sizes  = df[ask_size_cols].values    # [N, 10]
    bid_prices = df[bid_price_cols].values   # [N, 10]
    bid_sizes  = df[bid_size_cols].values    # [N, 10]

    # Stack all levels: [N, 20]
    prices = np.hstack([ask_prices, bid_prices])  # [N, 20]
    sizes  = np.hstack([ask_sizes, bid_sizes])    # [N, 20]
    sides  = np.hstack([np.zeros((N, 10)), np.ones((N, 10))])  # [N, 20]

    # Final tensor: [N, 20, 3]
    lob_array = np.stack([prices, sizes, sides], axis = -1)

    return torch.tensor(lob_array, dtype = torch.float32)

def compute_time_deltas(df, time_col = 'Time_dt', total_seconds = 19800):
    """
    total_seconds: only keep 10:00AM to 15:30PM data (default = 19800 seconds)
    
    Compute time deltas (in ratio) from opening time.
    Returns a float32 numpy array of shape [N]
    """
    base_time = df[time_col].iloc[0]
    deltas = (df[time_col] - base_time).dt.total_seconds().astype(np.float32).values
    ratios = deltas / total_seconds
    return ratios

class SlidingWindowDataset(Dataset):
    """
    Constructs a dataset using a sliding window approach. Each sample contains
    'past_window' data and 'predict_window' future data.
    Also computes the mid-price trend/volatility and liquidity/imbalance of the predict data.
    
    Args:
        data_tensor: Tensor of shape [N, 20, 3].
        time_deltas: Array of shape [N], time (in seconds) since market open.
        past_window: The number of past data points.
        predict_window: The number of future data points.
        step: Sliding window stride.
    """
    def __init__(self, data_tensor, time_deltas, past_window, predict_window, step = 1):
        self.data_tensor = data_tensor
        self.time_deltas = time_deltas
        self.past_window = past_window
        self.predict_window = predict_window
        self.step = step
        self.length = (len(data_tensor) - past_window - predict_window) // step + 1  # Number of sliding windows available
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Calculate the starting index based on the step size
        start = idx * self.step
        
        # [past_window, 20, 3]
        past_data = self.data_tensor[start: start + self.past_window]
        
        # [predict_window, 20, 3]
        future_data = self.data_tensor[start + self.past_window: start + self.past_window + self.predict_window]
        
        # reshape past_data and future_data to [predict_window, 20, 2] by abandoning
        # ask/bid sign colume (0/1), which does not affect training
        past_data   = past_data[:, :, :2]
        future_data = future_data[:, :, :2]
        
        # Compute the mid-price for last snapshot in past_data.
        # It is assumed that each data sample has shape [20, 2], where:
        # - The best ask price is located at row 9, column 0,
        # - The best bid price is located at row 10, column 0.
        mid_prices = []
        # Compute mid-price for the last snapshot in past_data
        last_past_data = past_data[-1].cpu()
        mid_prices.append((last_past_data[9, 0] + last_past_data[10, 0]) / 2)
        
        # Compute mid-price for each snapshot in future_data
        for d in future_data:
            best_ask = d[9, 0]
            best_bid = d[10, 0]
            mid_price = (best_ask + best_bid) / 2
            mid_prices.append(mid_price)
        mid_prices = torch.stack(mid_prices)  # Shape: [predict_window + 1]
        
        mid_prices_diff = torch.diff(mid_prices) # Shape: [predict_window]
        
        # Compute trend (log return) of the mid-prices
        # sum(x_t)
        trend = mid_prices_diff.sum() # Scalar, representing trend
        
        # Compute volatility (realized volatility) of the mid-prices
        # sum(x_t^{2})
        # volatility = torch.square(mid_prices_diff).sum()   # Scalar, representing volatility
        volatility = torch.std(mid_prices_diff)            # Scalar, representing volatility
        
        # Compute liquidity (limit order book depth) by volumes
        liquidity = torch.sum(future_data[:, :, 1], dim = 1) # shape = [predict_window]
        
        # Compute Orberbook Imbalance by volumes
        ask_liquidity = torch.sum(future_data[:, :10, 1], dim = 1)
        bid_liquidity = torch.sum(future_data[:, 10:, 1], dim = 1)
        oi = (ask_liquidity - bid_liquidity) / (ask_liquidity + bid_liquidity)

        # Time deltas for past + future
        past_time_slice = self.time_deltas[start : start + self.past_window]  # [self.past_window]
        past_time_deltas = torch.tensor(past_time_slice, dtype = torch.float32)
        
        future_time_slice = self.time_deltas[start + self.past_window: start + self.past_window + self.predict_window]  # [self.predict_window]
        future_time_deltas = torch.tensor(future_time_slice, dtype = torch.float32)
        
        # past_data shape: [past_window, 20, 2]
        # predict_data shape: [predict_window, 20, 2]
        # trend: scalar
        # volatility: scalar
        # liquidity shape: [predict_window]
        # oi shape: [predict_window]
        # past_time_deltas: [past_window + predict_window]
        # future_time_deltas: [predict_window]
        return past_data, future_data, trend, volatility, liquidity, oi, past_time_deltas, future_time_deltas


def get_dataloader(data_tensor, time_deltas, past_window, predict_window, batch_size, step = 1, shuffle = True):
    """
    Creates a DataLoader for the sliding window dataset.
    
    The custom collate function returns:
      - past_batch: a tensor of shape [batch_size, predict_window, 20, 2], the past data.
      - predict_batch: a tensor of shape [batch_size, predict_window, 20, 2], the future data.
      - trend_batch: a tensor of shape [batch_size], the trend values.
      - volatility_batch: a tensor of shape [batch_size], the volatility values.
      - liquidity_batch: a tensor of shape [batch_size, predict_window], the order book liquidity.
      - oi_batch: a tensor of shape [batch_size, predict_window], the order book imbalance.
    
    Args:
        past_window: Number of past data points.
        predict_window: Number of future data points.
        batch_size: Batch size for the DataLoader.
        step: The step size for sliding window sampling.
        shuffle: Whether to shuffle the data.
    """
    dataset = SlidingWindowDataset(data_tensor, time_deltas, past_window, predict_window, step)
    
    def collate_fn(batch):
        # Each batch element is a tuple:
        # (past_data, predict_data, volatility, trend, liquidity, past_time, predict_time)
        past_batch = torch.stack([item[0] for item in batch], dim = 0)         # [batch_size, past_window, 20, 2]
        predict_batch = torch.stack([item[1] for item in batch], dim = 0)      # [batch_size, predict_window, 20, 2]
        trend_batch = torch.stack([item[2] for item in batch])                 # [batch_size]
        volatility_batch = torch.stack([item[3] for item in batch])            # [batch_size]
        liquidity_batch = torch.stack([item[4] for item in batch], dim = 0)    # [batch_size, predict_window]
        oi_batch = torch.stack([item[5] for item in batch], dim = 0)           # [batch_size, predict_window]
        past_time_batch = torch.stack([item[6] for item in batch], dim = 0)    # [batch_size, past_window]
        predict_time_batch = torch.stack([item[7] for item in batch], dim = 0) # [batch_size, predict_window] 
        
        return past_batch, predict_batch, trend_batch, volatility_batch, liquidity_batch, oi_batch, past_time_batch, predict_time_batch
    
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, 
                            collate_fn = collate_fn, drop_last = True)
    return dataloader


def transform_past_predict_batch(past_batch, predict_batch):
    """
    Transform past_batch and predict_batch to:
      1) temporal mid-price returns
      2) cross-sectional price level returns
      3) original volume levels (unchanged)
      
    Args:
        past_batch: Tensor of shape [B, past_window, 20, 2]
        predict_batch: Tensor of shape [B, predict_window, 20, 2]
    
    Returns:
        Two tensors of shape [B, predict_window, 20, 2]
    """
    batch_size, past_window, num_levels, _ = past_batch.shape
    _, predict_window, _, _                = predict_batch.shape
    assert past_window == predict_window + 1, "past_window must be predict_window + 1"

    # compute mid-prices over future window
    past_mid = (past_batch[:, :, 9, 0] + past_batch[:, :, 10, 0]) / 2.0         # [B, past_window]
    future_mid = (predict_batch[:, :, 9, 0] + predict_batch[:, :, 10, 0]) / 2.0 # [B, predict_window]

    # temporal mid-price diffs for past: [B, past_window - 1]
    mid_diffs_past = past_mid[:, 1:] - past_mid[:, :-1]                         # diff return
    # mid_diffs_past = past_mid[:, 1:] / past_mid[:, :-1] - 1                   # pct return
    # mid_diffs_past = torch.log(past_mid[:, 1:]) - torch.log(past_mid[:, :-1]) # log return
    
    # temporal mid-price diffs for future: [B, predict_window]
    last_past_mid = past_mid[:, -1:].clone()          # [B, 1]
    mid_seq_future = torch.cat([last_past_mid, future_mid], dim = -1)
    mid_diffs_future = mid_seq_future[:, 1:] - mid_seq_future[:, :-1]                         # diff return
    # mid_diffs_future = mid_seq_future[:, 1:] / mid_seq_future[:, :-1] - 1                   # pct return
    # mid_diffs_future = torch.log(mid_seq_future[:, 1:]) - torch.log(mid_seq_future[:, :-1]) # log return

    # cross-sectional price-level diffs for past: [B, past_window - 1, num_levels - 1]
    past_prices = past_batch[..., 0]      # [past_window - 1, num_levels]
    past_price_diff = past_prices[:, 1:, :-1] - past_prices[:, 1:, 1:]
    
    # cross-sectional price-level diffs for future: [B, predict_window, num_levels - 1]
    future_prices = predict_batch[..., 0] # [predict_window, num_levels]
    future_price_diff = future_prices[:, :, :-1] - future_prices[:, :, 1:]

    # build a tensor for the price-derived feature
    price_feat_past   = torch.empty(batch_size, past_window - 1, num_levels, 1)
    price_feat_future = torch.empty(batch_size, predict_window, num_levels, 1)
    
    price_feat_past[:, :, 0, 0]   = mid_diffs_past
    price_feat_future[:, :, 0, 0] = mid_diffs_future
    
    price_feat_past[:, :, 1:, 0]   = past_price_diff
    price_feat_future[:, :, 1:, 0] = future_price_diff
    
    volume_feat_past = past_batch[:, 1:, :, -1:]     # [B, past_window - 1, num_levels, 1]
    volume_feat_future = predict_batch[:, :, :, -1:] # [B, past_window - 1, num_levels, 1]
    
    past_transformed   = torch.cat([price_feat_past, volume_feat_past], dim = -1)
    future_transformed = torch.cat([price_feat_future, volume_feat_future], dim = -1)
    
    # past_transformed: [B, past_window - 1, num_levels, 2]
    # predict_transformed: [B, predict_window, num_levels, 2]
    return past_transformed, future_transformed

def transform_sample_batch(past_batch, samples):
    """
    transform generated samples (temporal mid-price differences, cross-section price level differences) 
    to absolute price
    
    past_batch: Tensor of shape [B, past_window, 20, 2]
    samples: Tensor of shape [B, 1, 20, predict_window]

    Returns:
        future_prices: Tensor of shape [B, predict_window, 20, 1]
    """
    
    device = samples.device
    sample_batch_size, _, _, predict_window = samples.shape

    # Last snapshot from past: [B, 20, 2]
    last_past_data = past_batch[:, -1, :, :].to(device)

    # compute last mid price in batch, last_mid_price shape = [B]             
    last_mid_price = ((last_past_data[:, 9, 0] + last_past_data[:, 10, 0]) / 2)

    # compute price level in batch
    diff_mid = samples[:, 0, 0, :]              # [B, predict_window]
    diff_price_levels = samples[:, 0, 1:, :]    # [B, predict_window, 19]
    
    future_mid = last_mid_price + torch.cumsum(diff_mid, dim = 1)
    # future_mid = last_mid_price * torch.cumprod((1+diff_mid), dim = 1)
    # future_mid = last_mid_price * torch.exp(torch.cumsum(diff_mid, dim = 1))
    
    future_prices = []
    for t in range(predict_window):
        mid_t = future_mid[:, t]                 # [B]
        d = diff_price_levels[:, :, 19]          # [B, 19], all positive values
        # p shape = [batch_size, 20]
        p = torch.empty(sample_batch_size, 20, device = device, dtype = last_mid_price.dtype)

        # compute ask/bid 1 price
        p[:, 9] = mid_t + d[:, 9] / 2     # ask 1 price
        p[:, 10] = mid_t - d[:, 9] / 2    # bid 1 price

        # compute ask 2, ask 3, ..., ask 10 price
        for i in reversed(range(0, 9)):      # i = 8,7,...,0
            p[:, i] = p[:, i+1] + d[:, i]

        # compute bid 2, bid 3, ..., bid 10 price
        for i in range(11, 20):              # i = 11,...,19
            p[:, i] = p[:, i-1] - d[:, i-1]

        future_prices.append(p.unsqueeze(1))

    # future_prices shape = [B, predict_window, 1, 20]
    future_prices = torch.cat(future_prices, dim = 1).unsqueeze(-1)
    
    return future_prices
    

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))

def scaler(x):
    """to be done"""
    return x

def inverse_scaler(x):
    """to be done"""
    return x

# --------------------------------------------------------
# Transforming Return Series By Norm-Salvaged Embedding
# --------------------------------------------------------

def get_deterministic_basis(T, device = None, dtype = None):
    """
    Constructs the orthonormal basis (The Skeleton) used for the transformation.
    
    1. The first basis vector u1 is fixed and represents the 'Trend' direction.
    2. The remaining basis vectors u2...uT are generated via QR decomposition, 
       ensuring they are orthonormal and deterministic (like a stable Gram-Schmidt).
    
    Args:
        T (int): The length of the time series.
    
    Returns:
        torch.Tensor: An orthonormal matrix Q of shape (T, T), where columns are the basis vectors.
    """
    # 1. Define the trend basis vector u1 (normalized constant vector)
    # This is like fixing the 'North' direction on a map.
    u1 = torch.ones(T, device = device, dtype = dtype) / torch.sqrt(torch.tensor(T, device = device, dtype = dtype))
    
    # 2. Construct an auxiliary matrix A for generating the rest of the basis
    # We use QR decomposition for numerical stability.
    A = torch.eye(T, device=device, dtype=dtype)
    A[:, 0] = u1
    
    # Q is the orthonormal matrix; its columns are the basis vectors u1, u2, ..., uT
    Q, _ = torch.linalg.qr(A)
    
    # Correction: Ensure u1 in the matrix is non-negative (for consistency)
    if Q[0, 0] < 0:
        Q = -Q
        
    return Q

def nse_transform(x):
    """
    Performs the Norm-Salvaged Embedding (NSE) transformation.
    Converts a time series x into an embedding vector y.
    
    Args:
        x (torch.Tensor): The input time series (e.g., return series), shape (T,).
    
    Returns:
        torch.Tensor: The NSE embedding y, shape (T,). 
        Structure: [Trend_Component, Volatility_Norm, Angle_1, Angle_2, ...]
    """
    T = x.shape[0]
    device = x.device
    dtype = x.dtype
    
    basis = get_deterministic_basis(T, device, dtype)
    
    # --- Stage 1: Basis Transformation (Orthogonal Decomposition) ---
    # Project x onto the orthonormal basis to get coefficients
    coeffs = torch.matmul(basis.T, x)
    
    trend_component = coeffs[0]      # Projection onto u1 (The Trend)
    residual_vector = coeffs[1:]     # Projections onto u2...uT (The Residual 'r')
    
    # --- Stage 2: Polar Transformation (Extracting Norm and Shape) ---
    # 1. Extract the Norm (Volatility)
    norm = torch.norm(residual_vector)
    
    # 2. Extract the Angles (Shape/Diversity)
    # This involves high-dimensional spherical coordinate transformation
    angles = []
    current_r = residual_vector.clone()
    
    # Loop to calculate T-1 angles
    for i in range(T - 2):
        if i == T - 3:
            # The last angle must use arctan2 to preserve the sign (corresponds to j=T-2 in Eq 8)
            last_angle = torch.atan2(current_r[-1], current_r[-2])
            angles.append(last_angle)
        else:
            # Use arccos for the preceding angles (corresponds to 1 <= j <= T-3 in Eq 8)
            current_norm = torch.norm(current_r[i:])
            
            # Handle numerical edge cases (when vector magnitude is near zero)
            if current_norm < 1e-9:
                phi = torch.tensor(0.0, device = device, dtype = dtype)
            else:
                # Clip value to ensure it's in the valid range [-1.0, 1.0] for arccos
                val = current_r[i] / current_norm
                val = torch.clamp(val, -1.0, 1.0) 
                phi = torch.acos(val)
            
            angles.append(phi)
            
    # Convert list of tensors to a single tensor
    angles_tensor = torch.stack(angles)
    
    y = torch.cat([
        trend_component.unsqueeze(0), 
        norm.unsqueeze(0), 
        angles_tensor
    ])
    
    return y

def inverse_nse_transform(y):
    """
    Performs the Inverse Norm-Salvaged Embedding (Inverse NSE) transformation.
    Restores the time series x from the embedding vector y.
    
    Args:
        y (torch.Tensor): The NSE embedding vector, shape (T,).
    
    Returns:
        torch.Tensor: The reconstructed time series x, shape (T,).
    """
    T = y.shape[0]
    device = y.device
    dtype = y.dtype
    
    basis = get_deterministic_basis(T, device, dtype)
    
    # 1. Unpack the Embedding
    trend_component = y[0]  # Trend component
    norm = y[1]             # Volatility Norm (||r||)
    angles = y[2:]          # Shape Angles (theta)
    
    # 2. Inverse Polar Transformation: Reconstruct the residual vector 'r'
    # This involves reconstructing r components from the norm and angles
    r_reconstructed = torch.zeros(T - 1, device = device, dtype = dtype)
    
    # The process involves a cumulative product of sine terms (sin_product)
    sin_product = torch.tensor(1.0, device = device, dtype = dtype)
    
    for i in range(T - 3):
        # Component calculation: Norm * (sin_product) * cos(theta_i)
        r_reconstructed[i] = norm * sin_product * torch.cos(angles[i])
        sin_product *= torch.sin(angles[i])
    
    # Special handling for the last two components
    # The second to last component
    r_reconstructed[-2] = norm * sin_product * torch.cos(angles[-1])
    # The last component
    r_reconstructed[-1] = norm * sin_product * torch.sin(angles[-1])
    
    # 3. Inverse Basis Transformation: Combine trend and residual
    # x = (Trend_Component * u1) + (r_reconstructed * u2...uT)
    # basis[:, 0] is u1, basis[:, 1:] are u2 to uT
    x_reconstructed = (trend_component * basis[:, 0]) + (torch.matmul(basis[:, 1:], r_reconstructed))
    
    return x_reconstructed