# Codes from SAITS

import os

import h5py
import numpy as np
import torch
import torchcde

def window_truncate(feature_vectors, seq_len):
    """ Generate time series samples, truncating windows from time-series data with a given sequence length.
    Parameters
    ----------
    feature_vectors: time series data, len(shape)=2, [total_length, feature_num]
    seq_len: sequence length
    """
    start_indices = np.asarray(range(feature_vectors.shape[0] // seq_len)) * seq_len
    sample_collector = []
    for idx in start_indices:
        sample_collector.append(feature_vectors[idx: idx + seq_len])

    return np.asarray(sample_collector).astype('float32')


def random_mask(vector, artificial_missing_rate):
    """generate indices for random mask"""
    assert len(vector.shape) == 1
    indices = np.where(~np.isnan(vector))[0].tolist()
    indices = np.random.choice(indices, int(len(indices) * artificial_missing_rate),replace=False) # 중복 없이!
    return indices
 
def make_coeffs(data_dict):
    X = data_dict['X']
    mask = data_dict['mask']
    
    return torchcde.natural_cubic_coeffs(torch.from_numpy(X))

def add_artificial_mask(X, artificial_missing_rate, set_name):
    """ Add artificial missing values.
    Parameters
    ----------
    X: feature vectors
    artificial_missing_rate: rate of artificial missing values that are going to be create
    set_name: dataset name, train/val/test
    """
    sample_num, seq_len, feature_num = X.shape


    if set_name == 'train':
        mask = (~np.isnan(X)).astype(np.float32)
        X_filledWith0 = np.nan_to_num(X)
        empirical_mean_for_GRUD = np.sum(mask * X_filledWith0, axis=(0, 1)) / np.sum(mask, axis=(0, 1))

        coeffs = torchcde.natural_cubic_coeffs(torch.from_numpy(X.reshape([sample_num, seq_len, feature_num])))
        coeffs = np.array([coeff.numpy() for coeff in coeffs])

        data_dict = {
            'X': X.reshape([sample_num, seq_len, feature_num]),
            'empirical_mean_for_GRUD': empirical_mean_for_GRUD,
            'coeffs': coeffs
        }

    else:
        X = X.reshape(-1)
        indices_for_holdout = random_mask(X, artificial_missing_rate)
        X_hat = np.copy(X)
        X_hat[indices_for_holdout] = np.nan  # X_hat contains artificial missing values
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        # indicating_mask contains masks indicating artificial missing values
        indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32)

        coeffs = torchcde.natural_cubic_coeffs(torch.from_numpy(X_hat.reshape([sample_num, seq_len, feature_num])))
        coeffs = np.array([coeff.numpy() for coeff in coeffs])

        data_dict= {
            'X': X.reshape([sample_num, seq_len, feature_num]),
            'X_hat': X_hat.reshape([sample_num, seq_len, feature_num]),
            'missing_mask': missing_mask.reshape([sample_num, seq_len, feature_num]),
            'indicating_mask': indicating_mask.reshape([sample_num, seq_len, feature_num]),
            'coeffs': coeffs
            }

    return data_dict



def saving_into_h5(saving_dir, data_dict, classification_dataset):
    """ Save data into h5 file.
    Parameters 
    ----------
    saving_dir: path of saving dir
    data_dict: data dictionary containing train/val/test sets
    classification_dataset: boolean, if this is a classification dataset
    """

    def save_each_set(handle, name, data):
        single_set = handle.create_group(name)
        if classification_dataset:
            # single_set.create_dataset('labels', data=data['labels'].astype(int))
            single_set.create_dataset('labels', data=data['labels'].astype('float'))
        single_set.create_dataset('X', data=data['X'].astype(np.float32))
        single_set.create_dataset('coeffs', data=data['coeffs'].astype(np.float32))
        if name in ['val', 'test']:
            single_set.create_dataset('X_hat', data=data['X_hat'].astype(np.float32))
            single_set.create_dataset('missing_mask', data=data['missing_mask'].astype(np.float32))
            single_set.create_dataset('indicating_mask', data=data['indicating_mask'].astype(np.float32))

    saving_path = os.path.join(saving_dir, 'datasets.h5')
    with h5py.File(saving_path, 'w') as hf:
        hf.create_dataset('empirical_mean_for_GRUD', data=data_dict['train']['empirical_mean_for_GRUD'])
        save_each_set(hf, 'train', data_dict['train'])
        save_each_set(hf, 'val', data_dict['val'])
        save_each_set(hf, 'test', data_dict['test'])
        print(saving_path)

