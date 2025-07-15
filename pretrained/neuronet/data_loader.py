# -*- coding:utf-8 -*-
import mne
import torch
import tqdm
import random
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


class TorchDataset(Dataset):
    def __init__(self, paths, sfreq, rfreq, ch_idx, scaler: bool = False, downsampling: bool = False):
        super().__init__()
        self.x, self.y = self.get_data(paths, sfreq, rfreq, ch_idx, scaler, downsampling)
        self.x, self.y = torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long)

    @staticmethod
    def get_data(paths, sfreq, rfreq, ch_idx, scaler_flag, downsampling):
        info = mne.create_info(sfreq=sfreq, ch_types='eeg', ch_names=['Fp1'])
        scaler = mne.decoding.Scaler(info=info, scalings='median')
        total_x, total_y = [], []
        for path in tqdm.tqdm(paths):
            data = np.load(path)
            x, y = data['x'], data['y']

            try:
                x = x[:, ch_idx, :].squeeze()
            except IndexError:
                continue

            if np.isnan(x).any():
                continue

            x = np.expand_dims(x, axis=1)
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

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = torch.tensor(self.x[item])
        y = torch.tensor(self.y[item])
        return x, y
