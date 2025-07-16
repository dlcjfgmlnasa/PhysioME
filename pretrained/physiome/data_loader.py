# -*- coding:utf-8 -*-
import os
import mne
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
    def __init__(self, paths, ch_names, sfreq: int, rfreq: int, scaler: bool = False, downsampling: bool = False):
        super().__init__()
        self.paths = paths
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.total_x, self.total_y = self.get_data(paths, ch_names, sfreq, rfreq, scaler, downsampling)

    def __len__(self):
        return self.total_x.shape[0]

    @staticmethod
    def get_data(paths, ch_names, sfreq, rfreq, scaler_flag, downsampling):
        info = mne.create_info(sfreq=sfreq, ch_types='eeg', ch_names=ch_names)
        scaler = mne.decoding.Scaler(info=info, scalings='median')

        total_x, total_y = [], []
        for path in tqdm.tqdm(paths):
            data = np.load(path)
            x, y = data['x'], data['y']

            if x.ndim != len(ch_names):
                continue

            if np.isnan(x).any():
                continue

            if scaler_flag:
                x = scaler.fit_transform(x)

            x = mne.EpochsArray(x, info=info)
            x = x.resample(rfreq)
            x = x.get_data().squeeze()

            total_x.append(x)
            total_y.append(y)
        total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)

        if downsampling:
            class_indices = {label: np.where(total_y == label)[0] for label in np.unique(total_y)}
            min_count = min(len(idxs) for idxs in class_indices.values())

            selected_indices = []
            for label, idxs in class_indices.items():
                sampled = np.random.choice(idxs, min_count, replace=False)
                selected_indices.extend(sampled)
            selected_indices = np.array(selected_indices)
            np.random.shuffle(selected_indices)
            total_x, total_y = total_x[selected_indices], total_y[selected_indices]
        return total_x, total_y

    def __getitem__(self, item):
        batch_x = torch.tensor(self.total_x[item], dtype=torch.float32)
        batch_y = torch.tensor(self.total_y[item], dtype=torch.int32)
        return batch_x, batch_y

