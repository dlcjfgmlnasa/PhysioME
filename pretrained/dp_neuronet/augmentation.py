# -*- coding:utf-8 -*-
# --------------------------------------------------------
# References:
# - https://github.com/belita0152/SimCLR/blob/main/dataset/augmentation.py
# - https://github.com/belita0152/Physiological_Signal_Segmentation/blob/main/pretrained/augmentation.py
# --------------------------------------------------------

import copy
import torch
import random
import torch.nn.functional as f


class DataAugmentationNeuroNet(object):
    def __init__(self, prob):
        self.prob = prob

    @staticmethod
    def transform_augmentation_1(x: torch.Tensor, p=0.5):
        input_length = x.shape[-1]
        segment_size = torch.randint(int(input_length / 5), int(input_length / 3),
                                     (1,), device=x.device).squeeze().item()
        if random.random() < p:
            index_1 = torch.randint(0, input_length - segment_size + 1, (1,), device=x.device).squeeze().item()
            index_2 = int(index_1 + segment_size)
            x_split = x[:, :, index_1:index_2]
            x_split = f.interpolate(x_split, size=input_length, mode='linear', align_corners=False)
        else:
            x_split = x
        return x_split

    @staticmethod
    def transform_augmentation_2(x: torch.Tensor, p=0.5):
        n_permutation = 5
        input_length = x.shape[-1]

        if random.random() < p:
            indexes = torch.randperm(input_length)[:n_permutation - 1]
            indexes = torch.cat([torch.tensor([0]), indexes, torch.tensor([input_length])])
            indexes, _ = torch.sort(indexes)

            segments = []
            for idx_1, idx_2 in zip(indexes[:-1], indexes[1:]):
                segments.append(x[:, :, idx_1:idx_2])

            perm_indices = torch.randperm(n_permutation)
            shuffled_segments = [segments[idx] for idx in perm_indices]
            x_split = torch.cat(shuffled_segments, dim=-1)
        else:
            x_split = x
        return x_split

    def __call__(self, x: torch.Tensor):
        x1 = self.transform_augmentation_1(x, p=self.prob)
        x2 = self.transform_augmentation_2(x, p=self.prob)
        return x1.squeeze(), x2.squeeze()

