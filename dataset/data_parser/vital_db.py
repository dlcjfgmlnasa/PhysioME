# -*- coding:utf-8 -*-
import os
import tqdm
import vitaldb
import numpy as np
import matplotlib.pyplot as plt


def extract_sample(data, sfreq: int = 100, duration: int = 60):
    # Extracts valid 30-second input segments and determines whether hypotension occurs 5 minutes later.
    length = data.shape[0]
    minutes_ahead = 2       # 2 minute

    total_x, total_y = [], []
    for i in range(0, length - sfreq * (duration + (1 + minutes_ahead) * 60), duration * sfreq):
        segx = data[i: i + sfreq * duration]
        segy = data[i + sfreq * (duration + minutes_ahead * 60):
                    i + sfreq * (duration + (minutes_ahead + 1) * 60)]

        # Validity checks
        if (
            np.isnan(segx).mean() > 0.1 or
            np.isnan(segy).mean() > 0.1 or
            (segx.T[0] < 10).any() or (segx.T[0] > 200).any() or
            (segy.T[0] < 10).any() or (segy.T[0] > 200).any() or
            (segx.T[1] < -5).any() or (segx.T[1] > 5).any() or
            (segy.T[1] < -5).any() or (segy.T[1] > 5).any() or
            (np.abs(np.diff(segx.T[1])) > 1.0).any() or
            (np.abs(np.diff(segy.T[1])) > 1.0).any()
        ):
            continue

        # Calculate moving average MAP and detect hypotension event
        n = 2 * sfreq
        abp_c = np.nancumsum(segy.T[0], dtype=np.float32)
        abp_c[n:] = abp_c[n:] - abp_c[:-n]
        abp_c = abp_c[n - 1:] / n
        event = np.nanmax(abp_c) < 65  # MAP < 65 mmHg

        total_x.append(segx.T)
        total_y.append([int(event)])

    if total_y:
        total_x = np.stack(total_x).squeeze()
        total_y = np.stack(total_y).squeeze()
        return total_x, total_y
    else:
        return None, None


def vitaldb_converter(src_path, trg_path, sfreq=100, duration=60):
    # Converts VitalDB files to input-target pairs and saves them as npz files.
    modal_names = ['SNUADC/ART', 'SNUADC/ECG_II', 'SNUADC/PLETH']

    def is_all_in(a, b):
        return all(item in a for item in b)

    os.makedirs(trg_path, exist_ok=True)
    paths = sorted(os.listdir(src_path))

    for fname in tqdm.tqdm(paths):
        subject_idx = os.path.splitext(fname)[0]
        full_path = os.path.join(src_path, fname)

        subject_modal_names = vitaldb.vital_trks(full_path)
        if is_all_in(subject_modal_names, modal_names):
            data = vitaldb.vital_recs(full_path, modal_names, 1 / sfreq)
            x, y = extract_sample(data, sfreq=sfreq, duration=duration)

            if x is not None:
                np.savez(os.path.join(trg_path, subject_idx + '.npz'), x=x, y=y)


if __name__ == '__main__':
    vitaldb_converter(
        src_path=os.path.join('..', '..', '..', '..', 'Dataset', 'vitaldb'),
        trg_path=os.path.join('..', '..', 'data', 'vitaldb_60sec_5min')
    )
