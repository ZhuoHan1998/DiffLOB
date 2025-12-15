import glob
import re
import datetime
import pandas as pd
import numpy as np
import os
import pickle
from datetime import time

def read_messages_orderbook(message_file, orderbook_file):
    raw_messages = pd.read_csv(message_file, 
                           header = None, names = ['Time', 'Event_Type', 'Order_ID', 'Size', 'Price', 'Direction'])
    raw_orderbook = pd.read_csv(orderbook_file,
                           header = None, names = ['Ask_Price_1', 'Ask_Size_1', 'Bid_Price_1', 'Bid_Size_1',
                                                   'Ask_Price_2', 'Ask_Size_2', 'Bid_Price_2', 'Bid_Size_2',
                                                   'Ask_Price_3', 'Ask_Size_3', 'Bid_Price_3', 'Bid_Size_3',
                                                   'Ask_Price_4', 'Ask_Size_4', 'Bid_Price_4', 'Bid_Size_4',
                                                   'Ask_Price_5', 'Ask_Size_5', 'Bid_Price_5', 'Bid_Size_5',
                                                   'Ask_Price_6', 'Ask_Size_6', 'Bid_Price_6', 'Bid_Size_6',
                                                   'Ask_Price_7', 'Ask_Size_7', 'Bid_Price_7', 'Bid_Size_7',
                                                   'Ask_Price_8', 'Ask_Size_8', 'Bid_Price_8', 'Bid_Size_8',
                                                   'Ask_Price_9', 'Ask_Size_9', 'Bid_Price_9', 'Bid_Size_9',
                                                   'Ask_Price_10', 'Ask_Size_10', 'Bid_Price_10', 'Bid_Size_10',])
    return raw_messages, raw_orderbook

def resample_orderbook(raw_messages, raw_orderbook):
    raw_orderbook["Time"] = raw_messages["Time"]
    raw_orderbook['Time_dt'] = pd.to_datetime(raw_orderbook['Time'], unit = 's')
    
    # abandon first 30mins and last 30mins because of volatility
    # only keep 10:00AM to 15:30PM data
    raw_orderbook = raw_orderbook[(raw_orderbook['Time_dt'].dt.time >= time(10, 0)) & (raw_orderbook['Time_dt'].dt.time <= time(15, 30))]
    
    raw_orderbook.set_index('Time_dt', inplace = True)
    
    # get a sample every 1 second
    raw_orderbook_sampled = raw_orderbook.resample('1S').first().reset_index()

    # divide all price columes by 10000, transform the price unit to dollar
    price_cols = [col for col in raw_orderbook_sampled.columns if "Price" in col]
    raw_orderbook_sampled[price_cols] = raw_orderbook_sampled[price_cols] / 10000
    
    raw_orderbook_sampled = raw_orderbook_sampled.dropna() # very few rows have Nan values, drop them
    
    raw_orderbook_sampled = clip_volume(raw_orderbook_sampled) # clip big volume values

    # return sampled orderbook
    return raw_orderbook_sampled

def process_folder(folder_path, split_rate):
    
    def extract_date(filename):
        """extract 'YYYY-MM-DD'
        """
        match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if match:
            return datetime.datetime.strptime(match.group(1), '%Y-%m-%d')

        return datetime.datetime.min

    # get all message files via glob
    message_files = glob.glob(os.path.join(folder_path, '*_message_*.csv'))
    
    # reorder message files
    message_files = sorted(message_files, key = extract_date)
    
    train_file_num = round(len(message_files) * split_rate[0])
    val_file_num = round(len(message_files) * split_rate[1])
    test_file_num = len(message_files) - train_file_num - val_file_num

    train_message_files = message_files[0 : train_file_num]
    val_message_files   = message_files[train_file_num : train_file_num + val_file_num]
    test_message_files  = message_files[train_file_num + val_file_num : train_file_num + val_file_num + test_file_num]
    
    def process_message_files(message_files):
        processed_results = []
        for message_file in message_files:
            orderbook_file = message_file.replace("message", "orderbook")
            if os.path.exists(orderbook_file):
                raw_messages, raw_orderbook = read_messages_orderbook(message_file, orderbook_file)
                processed_orderbook = resample_orderbook(raw_messages, raw_orderbook)
                processed_results.append(processed_orderbook)
            else:
                print(f"file doesn't exist: {orderbook_file}")
        return processed_results

    # process train, val, test separately
    train_results = process_message_files(train_message_files)
    val_results  =  process_message_files(val_message_files)
    test_results =  process_message_files(test_message_files)
    
    return train_results, val_results, test_results

def clip_volume(df):
    """Cap volume by 99 quantile
    """
    ask_size_cols  = [f"Ask_Size_{i}" for i in range(1, 11)]
    bid_size_cols  = [f"Bid_Size_{i}" for i in range(1, 11)]
    volume_cols = ask_size_cols + bid_size_cols
    
    for col in volume_cols:
        upper_threshold = df[col].quantile(0.99) 
        df[col] = np.clip(df[col], None, upper_threshold)
    
    return df
