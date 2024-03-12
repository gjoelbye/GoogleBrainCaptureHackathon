
import mne

import numpy as np
from tqdm.notebook import tqdm
import torch

from utils.deep1010 import to_deep1010, to_1020

def min_max_normalize(x: torch.Tensor, low=-1, high=1):

    xmin = x.min()
    xmax = x.max()
    
    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    return (high - low) * x

def add_scaling_channel(x: torch.Tensor, data_min = -0.001, data_max = 0.001, scale_idx = -1):
    
    X = np.zeros((x.shape[0] + 1, x.shape[1]))
    X[:-1] = x

    max_scale = data_max - data_min

    scale = 2 * (torch.clamp_max((x.max() - x.min()) / max_scale, 1.0) - 0.5)
    X[scale_idx] = scale

    return X

def deep1010_stuff(raw, channel_order):
    raw = raw.copy()

    X = raw.get_data()
    X = min_max_normalize(torch.tensor(X))
    X = add_scaling_channel(X)

    ch_names = channel_order + ['scaling']
    ch_types = ['eeg'] * len(channel_order) + ['misc']
    new_info = mne.create_info(ch_names, raw.info['sfreq'], ch_types=ch_types)

    
    new_raw = mne.io.RawArray(X, new_info, verbose=False)
    new_raw.set_montage(raw.get_montage())
    new_raw.set_meas_date(raw.info['meas_date'])
    new_raw.set_annotations(raw.annotations)
    new_raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)

    return new_raw

def pick_rename_reorder_channels(raw, channel_picks, channel_order):
    """Picks and renames the channels of the raw object.
    Parameters
    ----------
    raw : mne.io.Raw
        The raw object.
    config : list
        The configuration of the channels.
    Returns
    -------
    raw : mne.io.Raw
        The raw object.
    """
    raw.copy()
    
    raw.pick(picks=channel_picks['original'])
    raw.rename_channels({channel_picks['original'][i]: channel_picks['renamed'][i] for i in range(len(channel_picks['original']))})
    raw.reorder_channels(channel_order)

    return raw


def get_raw(edf_file_path: str,
            channel_picks: list, channel_order: list,
            preprocessing: bool = True, filter: bool = True, 
            resample = 256, high_pass = 0.5, low_pass = 70, notch = 60,
            montage = mne.channels.make_standard_montage('standard_1020')) -> mne.io.Raw:
    """Reads and preprocesses an EDF file.
    Parameters
    ----------
    edf_file_path : str
        The path to the EDF file.
    channel_config : list
        The channel configuration.
    filter : bool
        Whether to filter the data.
    high_pass : float
        The high pass frequency.
    low_pass : float
        The low pass frequency.
    notch : float  
        The notch frequency.    
    resample : int
        The resample frequency.
    montage : mne.channels.Montage
        The montage.
    Returns
    -------
    raw : mne.io.Raw
        The raw object.
    """
    raw = mne.io.read_raw(edf_file_path, preload=True, verbose='error')

    if not preprocessing:
        return raw
    
    raw = pick_rename_reorder_channels(raw, channel_picks, channel_order)

    raw = raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
    raw = raw.set_montage(montage)

    if filter:
        raw = raw.resample(resample, verbose=False)
        raw = raw.filter(high_pass, low_pass, fir_design='firwin', verbose=False)
        raw = raw.notch_filter(notch, fir_design='firwin', verbose=False)

    # TODO: remove deep1010 when ready
    raw = deep1010_stuff(raw, channel_order)
    
    return raw


def get_annotations(edf_file_path: str, window_length = None) -> mne.Annotations:
    """Reads an edf file and returns the annotations.
    Parameters
    ----------
    edf_file_path : str
        Path to the edf file.
    window_length : float
        The length of the window to split the annotations into.
    Returns
    -------
    annotations : mne.Annotations
        The annotations.
    """

    annotations = mne.read_annotations(edf_file_path)
    
    if isinstance(window_length, (int, float)):
        new_onset = []
        new_duration = []
        new_description = []

        for i in range(len(annotations)):  
            for j in range(int(annotations.duration[i] // window_length)):
                new_onset.append(annotations.onset[i] + j * window_length)
                new_duration.append(window_length)
                new_description.append(annotations.description[i])
                
        new_onset = np.array(new_onset, dtype=np.float64)
        new_duration = np.array(new_duration, dtype=np.float64)
        new_description = np.array(new_description, dtype='<U2')

        annotations = mne.Annotations(onset=new_onset, duration=new_duration, description=new_description)

    return annotations