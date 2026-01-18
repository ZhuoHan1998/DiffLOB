import pandas as pd
import numpy as np
import random
import argparse
import torch

from utils.util import load_config, convert_dataframe_to_tensor, compute_time_deltas, get_dataloader
from utils.preprocessing import process_folder

from autoregressive_trainer import AutoregressiveTrainer
from autoregressive_sampler import AutoregressiveSampler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "train or sample using autoregressive S5 config file")
    parser.add_argument("-c", "--config", type = str, required = True, help = "path of config file")
    args = parser.parse_args()

    user_cfg = load_config(args.config)
    config = user_cfg.Configuration()

    fix_seed = 100
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    train_orderbooks, val_orderbooks, test_orderbooks = process_folder(config.folder_path, config.split_rate)

    train_orderbooks = pd.concat(train_orderbooks, ignore_index=True)
    val_orderbooks = pd.concat(val_orderbooks, ignore_index=True)
    test_orderbooks = pd.concat(test_orderbooks, ignore_index=True)

    train_tensor = convert_dataframe_to_tensor(train_orderbooks)
    val_tensor = convert_dataframe_to_tensor(val_orderbooks)
    test_tensor = convert_dataframe_to_tensor(test_orderbooks)

    train_time = compute_time_deltas(train_orderbooks)
    val_time = compute_time_deltas(val_orderbooks)
    test_time = compute_time_deltas(test_orderbooks)

    past_window = config.past_window
    predict_window = config.predict_window
    store_length = config.store_length

    if config.is_training:
        train_loader = get_dataloader(
            train_tensor, train_time,
            past_window, predict_window,
            config.training_batch_size, store_length,
            shuffle = True
        )
        val_loader = get_dataloader(
            val_tensor, val_time,
            past_window, predict_window,
            config.training_batch_size, store_length,
            shuffle = True
        )

        trainer = AutoregressiveTrainer(train_dataloader = train_loader, val_dataloader = val_loader, config = config)
        trainer.train()
    else:
        sample_loader = get_dataloader(
            test_tensor, test_time,
            past_window, predict_window,
            config.sampling_batch_size, store_length,
            shuffle = False
        )

        sampler = AutoregressiveSampler(sample_dataloader = sample_loader, config = config)
        sampler.sample()
