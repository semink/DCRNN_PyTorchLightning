import torch
import os
from pytorch_lightning import LightningDataModule
import numpy as np
from lib import utils
from torch.utils.data import DataLoader


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, scaler, t_features, seq_len=1, horizon=1):
        self.X = X
        self.seq_len = seq_len
        self.horizon = horizon
        self.scaler = scaler
        self.t_features = t_features
        self.num_sensors = X.shape[1]

    def __len__(self):
        return self.X.__len__() - (self.seq_len + self.horizon)

    def get_data(self, tidx):
        x_time_index = slice(tidx, tidx + self.seq_len, 1)
        y_time_index = slice(tidx + self.seq_len, tidx +
                             self.seq_len + self.horizon, 1)
        traffic_X = self.scaler.transform(self.X[x_time_index, :].T)
        time_feature_X = np.tile(
            self.t_features[x_time_index].T, (self.num_sensors, 1))

        traffic_Y = self.scaler.transform(self.X[y_time_index, :].T)

        input = torch.stack([torch.tensor(feature) for feature in (
            traffic_X, time_feature_X)], dim=0).float()
        d_input = torch.tensor(traffic_Y).unsqueeze(0).float()
        return (input, d_input)

    def __getitem__(self, index):
        X, Y = self.get_data(index)
        return X.numpy(), Y.numpy()


class DataModule(LightningDataModule):
    def __init__(self, dataset: str = "bay", batch_size: int = 32,
                 seq_len=24, horizon=12, num_workers=1):
        super(DataModule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_workers = num_workers
        self.train_df, self.valid_df, self.test_df = None, None, None
        self.scaler = None
        self.dt = None
        self.adj = None
        self.train_t, self.valid_t, self.test_t = None, None, None

    def prepare_data(self, custom_dataset=None, residual=False):
        if custom_dataset is None:
            self.df, self.adj = utils.get_traffic_data(self.dataset)

            self.t_features = utils.convert_timestamp_to_feature(self.df.index)
        else:
            self.df, self.t_features, self.adj = custom_dataset

        # dt
        self.dt = self.df.index[1] - self.df.index[0]

        # split dataset to train/val/test
        self._split_train_val_test()

        # if residual
        if residual:
            self._set_residual()

        # set scaler
        self._set_scaler()

    def get_scaler(self):
        return self.scaler

    def get_num_nodes(self):
        return self.df.shape[1]

    def get_adj(self):
        return torch.tensor(self.adj).float()

    def _split_train_val_test(self):
        self.train_df, self.valid_df, self.test_df = utils.split_data(
            df=self.df)

        self.train_t, self.valid_t, self.test_t = utils.split_data(
            df=self.t_features)

    def _set_scaler(self):
        self.scaler = utils.StandardScaler(
            self.train_df.values.mean(), self.train_df.values.std())

    def get_raw_data(self):
        return self.df, self.adj

    def _cache_check(self, fn):
        return os.path.isfile(fn)

    def train_dataloader(self):
        data = self.train_df.values
        t = self.train_t.values
        ds = TimeSeriesDataset(data, scaler=self.scaler, t_features=t,
                               seq_len=self.seq_len, horizon=self.horizon)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        data = self.valid_df.values
        t = self.valid_t.values
        ds = TimeSeriesDataset(data, scaler=self.scaler, t_features=t,
                               seq_len=self.seq_len, horizon=self.horizon)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        data = self.test_df.values
        t = self.test_t.values
        ds = TimeSeriesDataset(data, scaler=self.scaler, t_features=t,
                               seq_len=self.seq_len, horizon=self.horizon)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
