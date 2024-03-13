
import mne
import torch
import numpy as np
from tqdm.notebook import tqdm

from src.data.conf.eeg_channel_picks import hackathon 
from src.data.conf.eeg_channel_order import standard_19_channel


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

    # TODO: Confirm compatibility between channel_picks['renamed'] and channel_order
    raw.reorder_channels(channel_order)

    return raw


def get_raw(edf_file_path: str,
            channel_picks: list = hackathon, channel_order: list = standard_19_channel,
            preprocessing: bool = True, filter: bool = True, 
            resample = 256, highpass = 1, lowpass = 70, notch = 60,
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
        raw = raw.filter(highpass, lowpass, fir_design='firwin', verbose=False)
        raw = raw.notch_filter(notch, fir_design='firwin', verbose=False)

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