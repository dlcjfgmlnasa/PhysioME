# -*- coding:utf-8 -*-
import os
import mne
import glob
import numpy as np


def sleep_physionet_converter(src_path, trg_path, duration=30):
    # Physionet Sleep Dataset Converter (Sleep-EDF expanded-1.0.0)
    # We used EEG Fpz-Cz channels
    # * Input  : Physionet Sleep Dataset (.edf)
    # * Output : Converted Dataset (.npy)

    ch_names = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
    annotation_desc_2_event_id = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4,
    }

    # create a new event_id that unifies stages 3 and 4
    event_id_1 = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3/4': 3,
        'Sleep stage R': 4,
    }

    event_id_2 = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage R': 4,
    }

    psg_fnames = glob.glob(os.path.join(src_path, '*PSG.edf'))
    ann_fnames = glob.glob(os.path.join(src_path, '*Hypnogram.edf'))
    psg_fnames.sort()
    ann_fnames.sort()

    for psg_fname, ann_fname in zip(psg_fnames, ann_fnames):
        total_x, total_y = [], []

        raw = mne.io.read_raw_edf(psg_fname, preload=True)
        ann = mne.read_annotations(ann_fname)

        # keep last 30-min wake events before sleep and first 30-min wake events after
        # sleep and redefine annotations on raw data
        ann.crop(ann[1]["onset"] - 30 * 60, ann[-2]["onset"] + 30 * 60)
        raw.set_annotations(ann, emit_warning=False)
        raw = raw.copy().pick(ch_names)
        raw = raw.copy().filter(0, 40)      # band pass filter (0 ~ 40Hz)

        event, _ = mne.events_from_annotations(
            raw=raw, event_id=annotation_desc_2_event_id, chunk_duration=duration
        )

        tmax = 30.0 - 1.0 / raw.info["sfreq"]
        try:
            epochs = mne.Epochs(raw=raw, events=event, event_id=event_id_1, tmin=0.0, tmax=tmax, baseline=None)
        except ValueError:
            epochs = mne.Epochs(raw=raw, events=event, event_id=event_id_2, tmin=0.0, tmax=tmax, baseline=None)

        for epoch, event in zip(epochs, epochs.events):
            total_x.append(epoch)
            total_y.append(event[-1])

        total_x = np.array(total_x).squeeze()
        total_y = np.array(total_y).squeeze()

        # Saving Numpy Array
        name = os.path.basename(psg_fname).split('-')[0].lower()
        np_path = os.path.join(trg_path, name)
        np.savez(np_path, x=total_x, y=total_y)


if __name__ == '__main__':
    sleep_physionet_converter(
        src_path=os.path.join('..', '..', '..', '..', 'Dataset', 'physionet.org',
                              'files', 'sleep-edfx', '1.0.0', 'sleep-cassette'),
        trg_path=os.path.join('..', '..', 'data', 'sleep_edfx')
    )
