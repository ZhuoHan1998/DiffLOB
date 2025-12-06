import pandas as pd
import numpy as np
import random
import argparse
import importlib.util
import torch

from utils.util import load_config, convert_dataframe_to_tensor, compute_time_deltas, get_dataloader
from utils.preprossing import process_folder
from diffusion_trainer import Trainer
from diffusion_sampler import Sampler


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "train or sample using DIFFUSION config file")
    parser.add_argument("-c", "--config", type = str, required = True, help = "path of config file")
    args = parser.parse_args()
    
    user_cfg = load_config(args.config)
    config = user_cfg.Configuration()
    
    fix_seed = 100
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # --------------------------------------------------------
    # preprocessing
    # --------------------------------------------------------

    # process all messages and orderbooks in given file
    train_orderbooks, val_orderbooks, test_orderbooks = process_folder(config.folder_path, config.split_rate)

    # get raw train, val, test orderbooks
    train_orderbooks = pd.concat(train_orderbooks, ignore_index = True)
    val_orderbooks = pd.concat(val_orderbooks, ignore_index = True)
    test_orderbooks = pd.concat(test_orderbooks, ignore_index = True)
    print(f"train orderbooks: {train_orderbooks.head()}",
          f"train orderbooks shape: {train_orderbooks.shape}")
    print(f"val orderbooks: {val_orderbooks.head()}",
          f"val orderbooks shape: {val_orderbooks.shape}")
    print(f"test orderbooks: {test_orderbooks.head()}",
          f"test orderbooks shape: {test_orderbooks.shape}")

    train_orderbooks_tensor = convert_dataframe_to_tensor(train_orderbooks)
    val_orderbooks_tensor = convert_dataframe_to_tensor(val_orderbooks)
    test_orderbooks_tensor = convert_dataframe_to_tensor(test_orderbooks)

    train_time_deltas = compute_time_deltas(train_orderbooks)
    val_time_deltas = compute_time_deltas(val_orderbooks)
    test_time_deltas = compute_time_deltas(test_orderbooks)
    
    past_window = config.past_window
    predict_window = config.predict_window
    training_batch_size = config.training_batch_size
    sampling_batch_size = config.sampling_batch_size
    store_length = config.store_length
    
    if config.is_training == True:
        
        # --------------------------------------------------------
        # training
        # --------------------------------------------------------

        # get training dataloader
        train_dataloader = get_dataloader(train_orderbooks_tensor, train_time_deltas, past_window, predict_window, training_batch_size, store_length, shuffle = True)
        val_dataloader = get_dataloader(val_orderbooks_tensor, val_time_deltas, past_window, predict_window, training_batch_size, store_length, shuffle = True)
        
        # training phase
        model_train = Trainer(train_dataloader = train_dataloader, val_dataloader = val_dataloader, config = config)
        model_train.train()
    
    else:

        # --------------------------------------------------------
        # sampling
        # --------------------------------------------------------

        # get sampling dataloader
        sample_dataloader = get_dataloader(test_orderbooks_tensor, test_time_deltas, past_window, predict_window, sampling_batch_size, store_length, shuffle = False)
        
        # sampling phase
        model_sampler = Sampler(sample_dataloader = sample_dataloader, config = config)
        model_sampler.sample()
    
    