# -*- coding:utf-8 -*-
import os
import numpy as np
from sklearn.model_selection import KFold
from scipy.signal import butter, lfilter


def butter_bandpass_filter(signal, low_cut, high_cut, fs, order=5):
    if low_cut == 0:
        low_cut = 0.5
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)
    return y


def group_cross_validation(base_path, test_size=0.25, holdout_subject_size: int = 10):
    paths = [os.path.join(base_path, path) for path in os.listdir(base_path)]
    size = len(paths)
    train_paths, val_paths = paths[:int(size * (1 - test_size))], paths[int(size * (1 - test_size)):]
    train_paths, eval_paths = train_paths[:len(train_paths) - holdout_subject_size], \
                             train_paths[len(train_paths) - holdout_subject_size:]

    print('[K-Group Cross Validation]')
    print('   >> Train Subject Size : {}'.format(len(train_paths)))
    print('   >> Validation Subject Size : {}'.format(len(val_paths)))
    print('   >> Evaluation Subject Size : {}\n'.format(len(eval_paths)))

    return {'train_paths': train_paths, 'val_paths': val_paths, 'eval_paths': eval_paths}