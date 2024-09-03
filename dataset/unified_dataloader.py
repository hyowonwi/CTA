import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import utils
import torchcde

# code from SAITS
model_type_list = ['VAE', 'AE', 'VAE_AE', 'AE_AE']
class LoadDataset(Dataset):
    def __init__(self, file_path, seq_len, feature_num, model_type):
        super(LoadDataset, self).__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.model_type = model_type


class LoadValTestDataset(LoadDataset):
    """Loading process of val or test set"""

    def __init__(self, file_path, set_name, seq_len, feature_num, model_type):
        super(LoadValTestDataset, self).__init__(file_path, seq_len, feature_num, model_type)
        with h5py.File(self.file_path, 'r') as hf:  # read data from h5 file
            self.X = hf[set_name]['X'][:]
            self.X_doubledot = hf[set_name]['X_hat'][:]
            self.observed_mask = hf[set_name]['missing_mask'][:]    # 1 if observed after intentionally removing data (Fig 4)
            self.removed_mask = hf[set_name]['indicating_mask'][:]  # 1 if intentionally removed (Fig 4)
            self.coeffs = hf[set_name]['coeffs'][:]

        self.X = np.nan_to_num(self.X)
        self.X_doubledot = np.nan_to_num(self.X_doubledot)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_type in model_type_list:
            sample = (
                torch.tensor(idx),
                torch.from_numpy(self.X_doubledot[idx].astype('float32')),
                torch.from_numpy(self.observed_mask[idx].astype('float32')),
                torch.from_numpy(self.X[idx].astype('float32')),
                torch.from_numpy(self.removed_mask[idx].astype('float32')),
                self.coeffs[idx],
            )
        else:
            assert ValueError, f'Error model type: {self.model_type}'
        return sample


class LoadTrainDataset(LoadDataset):
    """Loading process of train set"""

    def __init__(self, file_path, seq_len, feature_num, model_type, masked_imputation_task, artificial_missing_rate):
        super(LoadTrainDataset, self).__init__(file_path, seq_len, feature_num, model_type)
        self.masked_imputation_task = masked_imputation_task
        if masked_imputation_task:
            self.artificial_missing_rate = 0.2
            assert 0 < self.artificial_missing_rate < 1, 'artificial_missing_rate should be greater than 0 and less than 1'

        with h5py.File(self.file_path, 'r') as hf:  # read data from h5 file
            self.X = hf['train']['X'][:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]

        X = X.reshape(-1)
        indices = np.where(~np.isnan(X))[0].tolist()
        indices = np.random.choice(indices, round(len(indices) * self.artificial_missing_rate))
        X_doubledot = np.copy(X)
        X_doubledot[indices] = np.nan  # mask values selected by indices
        observed_mask = (~np.isnan(X_doubledot)).astype(np.float32)  # 1 if observed after intentionally removing data (Fig 4)
        removed_mask = ((~np.isnan(X)) ^ (~np.isnan(X_doubledot))).astype(np.float32) # 1 if intentionally removed (Fig 4)

        X = X.reshape(self.seq_len, self.feature_num)
        X_doubledot = X_doubledot.reshape(self.seq_len, self.feature_num)
        
        coeffs = torchcde.natural_cubic_coeffs(torch.from_numpy(X_doubledot))
        
        X = np.nan_to_num(X)
        X_doubledot = np.nan_to_num(X_doubledot)
        
        observed_mask = observed_mask.reshape(self.seq_len, self.feature_num)
        removed_mask = removed_mask.reshape(self.seq_len, self.feature_num)
        
        if self.model_type in model_type_list:
            sample = (
                torch.tensor(idx),
                torch.from_numpy(X_doubledot.astype('float32')),
                torch.from_numpy(observed_mask.astype('float32')),
                torch.from_numpy(X.astype('float32')),
                torch.from_numpy(removed_mask.astype('float32')),
                coeffs,
            )
        else:
            assert ValueError, f'Error model type: {self.model_type}'

        return sample


class UnifiedDataLoader:
    def __init__(self, dataset_path, seq_len, feature_num, model_type, batch_size=1024, num_workers=4,
                 masked_imputation_task=False):
        """
        dataset_path: path of directory storing h5 dataset;
        seq_len: sequence length, i.e. time steps;
        feature_num: num of features, i.e. feature dimensionality;
        batch_size: size of mini batch;
        num_workers: num of subprocesses for data loading;
        model_type: model type, determine returned values;
        masked_imputation_task: whether to return data for masked imputation task, only for training/validation sets;
        """
        self.dataset_path = os.path.join(dataset_path, 'datasets.h5')
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.masked_imputation_task = masked_imputation_task

        self.train_dataset, self.train_loader, self.train_set_size = None, None, None
        self.val_dataset, self.val_loader, self.val_set_size = None, None, None
        self.test_dataset, self.test_loader, self.test_set_size = None, None, None

    def get_train_val_dataloader(self, artificial_missing_rate=None):
        self.train_dataset = LoadTrainDataset(self.dataset_path, self.seq_len, self.feature_num, self.model_type,
                                              self.masked_imputation_task, artificial_missing_rate)
        self.val_dataset = LoadValTestDataset(self.dataset_path, 'val', self.seq_len, self.feature_num, self.model_type)
        self.train_set_size = self.train_dataset.__len__()
        self.val_set_size = self.val_dataset.__len__()
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return self.train_loader, self.val_loader

    def get_test_dataloader(self):
        self.test_dataset = LoadValTestDataset(self.dataset_path, 'test', self.seq_len, self.feature_num,
                                               self.model_type)
        self.test_set_size = self.test_dataset.__len__()
        self.test_loader = DataLoader(self.test_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return self.test_loader

    def prepare_dataloader_for_imputation(self, set_name):
        data_for_imputation = LoadDataForImputation(self.dataset_path, set_name, self.seq_len, self.feature_num,
                                                    self.model_type)
        dataloader_for_imputation = DataLoader(data_for_imputation, self.batch_size, shuffle=False)
        return dataloader_for_imputation

    def prepare_all_data_for_imputation(self):
        train_set_for_imputation = self.prepare_dataloader_for_imputation('train')
        val_set_for_imputation = self.prepare_dataloader_for_imputation('val')
        test_set_for_imputation = self.prepare_dataloader_for_imputation('test')
        return train_set_for_imputation, val_set_for_imputation, test_set_for_imputation
