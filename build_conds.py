import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from utils.util import convert_dataframe_to_tensor, compute_time_deltas, get_dataloader, transform_past_predict_batch
from utils.preprocessing import read_messages_orderbook, resample_orderbook, process_folder

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# get val and test real samples
train_orderbooks, val_orderbooks, test_orderbooks = process_folder("/scratch/prj/genai_techniques_finance/lobsterdata/202302/AMZN/output-2023-02/0/0/43/",
                                                                   (.85, .05, .10))

# get raw orderbooks
train_orderbooks = pd.concat(train_orderbooks, ignore_index = True)
val_orderbooks = pd.concat(val_orderbooks, ignore_index = True)
test_orderbooks = pd.concat(test_orderbooks, ignore_index = True)

train_orderbooks_tensor = convert_dataframe_to_tensor(train_orderbooks)
val_orderbooks_tensor = convert_dataframe_to_tensor(val_orderbooks)
test_orderbooks_tensor = convert_dataframe_to_tensor(test_orderbooks)

train_time_deltas = compute_time_deltas(train_orderbooks)
val_time_deltas = compute_time_deltas(val_orderbooks)
test_time_deltas = compute_time_deltas(test_orderbooks)

# get real samples
past_window = 33
predict_window = 32
batch_size = 1
store_length = 32
train_dataloader = get_dataloader(train_orderbooks_tensor, train_time_deltas, past_window, predict_window, batch_size, store_length, shuffle = False)
val_dataloader = get_dataloader(val_orderbooks_tensor, val_time_deltas, past_window, predict_window, batch_size, store_length, shuffle = False)
test_dataloader = get_dataloader(test_orderbooks_tensor, test_time_deltas, past_window, predict_window, batch_size, store_length, shuffle = False)

# train_real_samples = []
# for _, predict_batch, _, _, _, _, _ in train_dataloader:
#     train_real_samples.append(predict_batch)
# train_real_samples = torch.cat(train_real_samples, dim = 0)[:, :store_length, :, :].numpy()
# train_real_samples = train_real_samples.reshape(-1, *train_real_samples.shape[2:])

# val_real_samples = []
# for _, predict_batch, _, _, _, _, _ in val_dataloader:
#     val_real_samples.append(predict_batch)
# val_real_samples = torch.cat(val_real_samples, dim = 0)[:, :store_length, :, :].numpy()
# val_real_samples = val_real_samples.reshape(-1, *val_real_samples.shape[2:])

test_real_samples = []
test_real_trend_cond = []
test_real_volatility_cond = []
test_real_liquidity_cond = []
test_real_imb_cond = []
for _, predict_batch, trend_batch, volatility_batch, liquidity_batch, imb_batch, _, _ in test_dataloader:
    test_real_samples.append(predict_batch)
    test_real_trend_cond.append(trend_batch.squeeze(0))
    test_real_volatility_cond.append(volatility_batch.squeeze(0))
    test_real_liquidity_cond.append(liquidity_batch.squeeze(0))
    test_real_imb_cond.append(imb_batch.squeeze(0))
    
test_real_samples = torch.cat(test_real_samples, dim = 0)[:, :store_length, :, :].numpy()
test_real_samples = test_real_samples.reshape(-1, *test_real_samples.shape[2:])

# for trend and volatility
test_real_trend_cond = torch.stack(test_real_trend_cond).flatten().numpy()
test_real_volatility_cond = torch.stack(test_real_volatility_cond).flatten().numpy()

# for liquidity and imbalance
test_real_liquidity_cond = torch.cat(test_real_liquidity_cond, dim = 0).numpy()
test_real_imb_cond = torch.cat(test_real_imb_cond, dim = 0).numpy()

print(test_real_samples.shape)
print(test_real_trend_cond.shape)
print(test_real_volatility_cond.shape)
print(test_real_liquidity_cond.shape)
print(test_real_imb_cond.shape)


x_price_train = []
x_trend_train = []
x_vol_train = []
x_price_test = []
x_trend_test = []
x_vol_test = []
x_liq_test = []
x_imb_test = []
for past_batch, predict_batch, trend_batch, volatility_batch, _, _, _, _ in train_dataloader:
    transformed_past_batch, transformed_predict_batch = transform_past_predict_batch(past_batch, predict_batch, 10) 
    x_price_train.append(transformed_predict_batch[:, :, 0, 0])
    x_trend_train.append(trend_batch.squeeze(0))
    x_vol_train.append(volatility_batch.squeeze(0))
    
for past_batch, predict_batch, trend_batch, volatility_batch, liquidity_batch, imb_batch, _, _ in test_dataloader:
    transformed_past_batch, transformed_predict_batch = transform_past_predict_batch(past_batch, predict_batch, 10) 
    x_price_test.append(transformed_predict_batch[:, :, 0, 0])
    x_trend_test.append(trend_batch.squeeze(0))
    x_vol_test.append(volatility_batch.squeeze(0))
    x_liq_test.append(liquidity_batch.squeeze(0))
    x_imb_test.append(imb_batch.squeeze(0))


x_price_train = torch.cat(x_price_train, dim = 0).numpy()
x_trend_train = torch.stack(x_trend_train).flatten().numpy()
x_vol_train = torch.stack(x_vol_train).flatten().numpy()
x_price_test = torch.cat(x_price_test, dim = 0).numpy()
x_trend_test = torch.stack(x_trend_test).flatten().numpy()
x_vol_test = torch.stack(x_vol_test).flatten().numpy()
x_liq_test = torch.stack(x_liq_test).flatten().numpy()
x_imb_test = torch.stack(x_imb_test).flatten().numpy()


np.save("conds/AAPL/trend_cond.npy", x_trend_test)
np.save("conds/AAPL/volatility_cond.npy", x_vol_test)
np.save("conds/AAPL/liquidity_cond.npy", x_liq_test)
np.save("conds/AAPL/imb_cond.npy", x_imb_test)