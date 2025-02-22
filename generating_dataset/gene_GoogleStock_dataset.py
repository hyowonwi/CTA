# Codes from SAITS
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append('..')
from utils import setup_logger
from data_processing_utils import window_truncate, random_mask, add_artificial_mask, \
    saving_into_h5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate UCI electricity dataset')
    parser.add_argument("--file_path", help='path of dataset file', type=str)
    parser.add_argument("--artificial_missing_rate", help='artificially mask out additional values',
                        type=float, default=0.1)
    parser.add_argument("--seq_len", help='sequence length', type=int, default=100)
    parser.add_argument('--dataset_name', help='name of generated dataset, will be the name of saving dir', type=str,
                        default='test')
    parser.add_argument('--saving_path', type=str, help='parent dir of generated dataset', default='.')
    args = parser.parse_args()

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    logger = setup_logger(os.path.join(dataset_saving_dir + "/dataset_generating.log"),
                          'Generate GoogleStock dataset', mode='w')
    logger.info(args)

    df = pd.read_csv(args.file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    feature_names = df.columns.tolist()
    feature_num = len(feature_names)
    df['datetime'] = pd.to_datetime(df.index)

    unique_months = df['datetime'].dt.to_period('M').unique()
    selected_as_test = unique_months[:24]  # select first 10 months as test set
    logger.info(f'months selected as test set are {selected_as_test}')
    selected_as_val = unique_months[24:48]  # select the 11st - the 20th months as val set
    logger.info(f'months selected as val set are {selected_as_val}')
    selected_as_train = unique_months[48:]  # use left months as train set
    logger.info(f'months selected as train set are {selected_as_train}')
    test_set = df[df['datetime'].dt.to_period('M').isin(selected_as_test)]
    val_set = df[df['datetime'].dt.to_period('M').isin(selected_as_val)]
    train_set = df[df['datetime'].dt.to_period('M').isin(selected_as_train)]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_set_X = scaler.transform(val_set.loc[:, feature_names])
    test_set_X = scaler.transform(test_set.loc[:, feature_names])

    train_set_X = window_truncate(train_set_X, args.seq_len)
    val_set_X = window_truncate(val_set_X, args.seq_len)
    test_set_X = window_truncate(test_set_X, args.seq_len)

    # add missing values in train set manually
    if args.artificial_missing_rate > 0:
        train_set_X_shape = train_set_X.shape
        train_set_X = train_set_X.reshape(-1)
        indices = random_mask(train_set_X, args.artificial_missing_rate)
        train_set_X[indices] = np.nan
        train_set_X = train_set_X.reshape(train_set_X_shape)
        logger.info(f'Already masked out {args.artificial_missing_rate * 100}% values in train set')
    else:
        args.artificial_missing_rate = 0.1
        
    train_set_dict = add_artificial_mask(train_set_X, args.artificial_missing_rate, 'train')
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, 'val')
    test_set_dict = add_artificial_mask(test_set_X, args.artificial_missing_rate, 'test')
    logger.info(f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}')
    logger.info(f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}')

    processed_data = {
        'train': train_set_dict,
        'val': val_set_dict,
        'test': test_set_dict
    }
    train_sample_num = len(train_set_dict["X"])
    val_sample_num = len(val_set_dict["X"])
    test_sample_num = len(test_set_dict["X"])
    total_sample_num = train_sample_num + val_sample_num + test_sample_num
    logger.info(f'Feature num: {feature_num},\n'
                f'{train_sample_num} ({(train_sample_num / total_sample_num):.3f}) samples in train set\n'
                f'{val_sample_num} ({(val_sample_num / total_sample_num):.3f}) samples in val set\n'
                f'{test_sample_num} ({(test_sample_num / total_sample_num):.3f}) samples in test set\n')

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    logger.info(f'All done. Saved to {dataset_saving_dir}.')
