# -*- coding:utf-8 -*-
import os
import tqdm
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import warnings


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


class TorchDataset(Dataset):
    def __init__(self, paths, ch_names, sfreq: int = 100):
        super().__init__()
        self.paths = paths
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.total_x, self.total_y = self.get_data()

    def __len__(self):
        return self.total_x.shape[0]

    def get_data(self):
        total_x, total_y = [], []
        for path in tqdm.tqdm(self.paths):
            data = np.load(path)
            x, y = data['x'], data['y']
            if x.ndim != len(self.ch_names):
                continue

            if np.isnan(x).any():
                continue

            total_x.append(x)
            total_y.append(y)
        total_x = np.concatenate(total_x)
        total_y = np.concatenate(total_y)
        return total_x, total_y

    def __getitem__(self, item):
        batch_x = torch.tensor(self.total_x[item], dtype=torch.float32)
        batch_y = torch.tensor(self.total_y[item], dtype=torch.int32)
        return batch_x, batch_y

