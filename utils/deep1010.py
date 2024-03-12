# Copyright (c) 2020, SPOClab-ca
# All rights reserved.

from ast import Tuple
from collections import OrderedDict

import numpy as np
import torch
from mne import create_info
from mne.io import RawArray

# Copyright (c) 2020, SPOClab-ca
# All rights reserved.

_LEFT_NUMBERS = list(reversed(range(1, 9, 2)))
_RIGHT_NUMBERS = list(range(2, 10, 2))

_EXTRA_CHANNELS = 5

DEEP_1010_CHS_LISTING = [
    # EEG
    "NZ",
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F9", *["F{}".format(n) for n in _LEFT_NUMBERS], "FZ", *["F{}".format(n) for n in _RIGHT_NUMBERS], "F10",

    "FT9", "FT7", *["FC{}".format(n) for n in _LEFT_NUMBERS[1:]], "FCZ",
    *["FC{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "FT8", "FT10",
                                                                                                                                  
    "T9", "T7", "T3",  *["C{}".format(n) for n in _LEFT_NUMBERS[1:]], "CZ",
    *["C{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "T4", "T8", "T10",

    "TP9", "TP7", *["CP{}".format(n) for n in _LEFT_NUMBERS[1:]], "CPZ",
    *["CP{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "TP8", "TP10",

    "P9", "P7", "T5",  *["P{}".format(n) for n in _LEFT_NUMBERS[1:]], "PZ",
    *["P{}".format(n) for n in _RIGHT_NUMBERS[:-1]],  "T6", "P8", "P10",

    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1",  "OZ", "O2",
    "IZ",
    # EOG
    "VEOGL", "VEOGR", "HEOGL", "HEOGR",

    # Ear clip references
    "A1", "A2", "REF",
    # SCALING
    "SCALE",
    # Extra
    *["EX{}".format(n) for n in range(1, _EXTRA_CHANNELS+1)]
]

EEG_INDS = list(range(0, DEEP_1010_CHS_LISTING.index('VEOGL')))
EOG_INDS = [DEEP_1010_CHS_LISTING.index(ch) for ch in ["VEOGL", "VEOGR", "HEOGL", "HEOGR"]]
REF_INDS = [DEEP_1010_CHS_LISTING.index(ch) for ch in ["A1", "A2", "REF"]]
EXTRA_INDS = list(range(len(DEEP_1010_CHS_LISTING) - _EXTRA_CHANNELS, len(DEEP_1010_CHS_LISTING)))
SCALE_IND = DEEP_1010_CHS_LISTING.index("SCALE")
_NUM_EEG_CHS = len(EEG_INDS)

DEEP_1010_TYPES = ["eeg"] * len(EEG_INDS) + ["eog"] * len(EOG_INDS) + ["misc"] * len(REF_INDS) + ["misc"] * len(EXTRA_INDS) + ["misc"]




def create_channel_type_dict(raw):
    """
    Takes a .edf raw and returns a dictionary of channel names and types
    Type names are resolved with heuristic functions below
    """

    channel_names = raw.info['ch_names']
    channel_types = raw.get_channel_types()

    type_dict = dict()
    for k, v in dict(zip(channel_names, channel_types)).items():
        if any([x in k for x in ["A1", "A2"]]):
            type_dict[k] = 'ref'
        else:
            type_dict[k] = v

    return type_dict

def _valid_character_heuristics(name, informative_characters):
    possible = ''.join(c for c in name.upper() if c in informative_characters).replace(' ', '')
    if possible == "":
        # print("Could not use channel {}. Could not resolve its true label, rename first.".format(name))
        return None
    return possible

def _heuristic_eeg_resolution(eeg_ch_name: str):
    eeg_ch_name = eeg_ch_name.upper()
    # remove some common garbage
    eeg_ch_name = eeg_ch_name.replace('EEG', '')
    eeg_ch_name = eeg_ch_name.replace('REF', '')
    informative_characters = set([c for name in DEEP_1010_CHS_LISTING[:_NUM_EEG_CHS] for c in name])
    return _valid_character_heuristics(eeg_ch_name, informative_characters)


def _heuristic_eog_resolution(eog_channel_name):
    return _valid_character_heuristics(eog_channel_name, "VHEOGLR")

def _heuristic_ref_resolution(ref_channel_name: str):
    ref_channel_name = ref_channel_name.replace('EAR', '')
    ref_channel_name = ref_channel_name.replace('REF', '')
    if ref_channel_name.find('A1') != -1:
        return 'A1'
    elif ref_channel_name.find('A2') != -1:
        return 'A2'

    if ref_channel_name.find('L') != -1:
        return 'A1'
    elif ref_channel_name.find('R') != -1:
        return 'A2'
    return "REF"

def _heuristic_resolution(old_type_dict: OrderedDict):
    resolver = {'eeg': _heuristic_eeg_resolution, 
                'eog': _heuristic_eog_resolution, 
                'ref': _heuristic_ref_resolution,
                'extra': lambda x: x, None: lambda x: x}

    new_type_dict = OrderedDict()

    for old_name, ch_type in old_type_dict.items():
        if ch_type is None:
            new_type_dict[old_name] = None
            continue

        new_name = resolver[ch_type](old_name)
        if new_name is None:
            new_type_dict[old_name] = None
        else:
            while new_name in new_type_dict.keys():
                # print('Deep1010 Heuristics resulted in duplicate entries for {}, incrementing name, but will be lost '
                #       'in mapping'.format(new_name))
                new_name = new_name + '-COPY'
            new_type_dict[new_name] = old_type_dict[old_name]

    assert len(new_type_dict) == len(old_type_dict)
    return new_type_dict

def _check_num_and_get_types(type_dict):
    type_lists = list()
    for ch_type, max_num in zip(('eog', 'ref'), (len(EOG_INDS), len(REF_INDS))):
        channels = [ch_name for ch_name, _type in type_dict.items() if _type == ch_type]

        for name in channels[max_num:]:
            # print("Losing assumed {} channel {} because there are too many.".format(ch_type, name))
            type_dict[name] = None
        type_lists.append(channels[:max_num])
    return type_lists

def _apply_special_mapping(map, channel_names, type_lists):
    EOG, ear_ref = type_lists

    if isinstance(EOG, str):
        EOG = [EOG] * 4
    elif len(EOG) == 1:
        EOG = EOG * 4
    elif EOG is None or len(EOG) == 0:
        EOG = []
    elif len(EOG) == 2:
        EOG = EOG * 2
    else:
        assert len(EOG) == 4
    for eog_map, eog_std in zip(EOG, EOG_INDS):
        try:
            map[channel_names.index(eog_map), eog_std] = 1.0
        except ValueError:
            raise ValueError("EOG channel {} not found in provided channels.".format(eog_map))

    if isinstance(ear_ref, str):
        ear_ref = [ear_ref] * 2
    elif ear_ref is None:
        ear_ref = []
    else:
        assert len(ear_ref) <= len(REF_INDS)
    for ref_map, ref_std in zip(ear_ref, REF_INDS):
        try:
            map[channel_names.index(ref_map), ref_std] = 1.0
        except ValueError:
            raise ValueError("Reference channel {} not found in provided channels.".format(ref_map))
    
    extra_channels = [None for _ in range(_EXTRA_CHANNELS)]

    if isinstance(extra_channels, str):
        extra_channels = [extra_channels]
    elif extra_channels is None:
        extra_channels = []
    assert len(extra_channels) <= _EXTRA_CHANNELS
    for ch, place in zip(extra_channels, EXTRA_INDS):
        if ch is not None:
            map[channel_names.index(ch), place] = 1.0
    return map

def min_max_normalize(x: torch.Tensor, low=-1, high=1):
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    return (high - low) * x



def _deep_1010(map, names, eog, ear_ref, extra=[]):
    for i, ch in enumerate(names):
        if ch not in eog and ch not in ear_ref and ch not in extra:
            try:
                map[i, DEEP_1010_CHS_LISTING.index(str(ch).upper())] = 1.0
            except ValueError:
                # print("Warning: channel {} not found in standard layout. Skipping...".format(ch))
                continue

    # Normalize for when multiple values are mapped to single location
    summed = map.sum(axis=0)[np.newaxis, :]
    mapping = torch.from_numpy(np.divide(map, summed, out=np.zeros_like(map), where=summed != 0)).float()
    mapping.requires_grad_(False)
    return mapping

def mapping_to_deep1010(raw):
    channel_type_dict = create_channel_type_dict(raw)
    channel_names = list(channel_type_dict.keys())
    type_lists = _check_num_and_get_types(channel_type_dict)
    revised_channel_types = _heuristic_resolution(channel_type_dict)

    map = np.zeros((len(channel_names), len(DEEP_1010_CHS_LISTING)))
    map = _apply_special_mapping(map, channel_names, type_lists)

    EOG, ear_ref = type_lists
    mapping = _deep_1010(map, list(revised_channel_types.keys()), EOG, ear_ref)

    return mapping

def to_deep1010(raw, data_max, data_min):
    mapping = mapping_to_deep1010(raw)
    
    # From BENDR config files
    # https://github.com/SPOClab-ca/BENDR/blob/ac918abaec111d15fcaa2a8fcd2bd3d8b0d81a10/configs/pretraining.yml#L45
    # data_max = 3276.7
    # data_min = -1583.9258304722666
    if isinstance(data_max, tuple) and len(data_max) == 1:
        data_max = data_max[0]
    if isinstance(data_min, tuple) and len(data_min) == 1:
        data_min = data_min[0]

    max_scale = data_max - data_min

    x = torch.tensor(raw.get_data())
    x = (x.transpose(1, 0) @ mapping.double()).transpose(1, 0)

    for ch_type_inds in (EEG_INDS, EOG_INDS, REF_INDS, EXTRA_INDS):
        x[ch_type_inds, :] = min_max_normalize(x[ch_type_inds, :])

    used_channel_mask = mapping.sum(dim=0).bool()
    x[~used_channel_mask, :] = 0

    scale = 2 * (torch.clamp_max((x.max() - x.min()) / max_scale, 1.0) - 0.5)
    x[SCALE_IND, :] = scale
    
    x = x.numpy()

    new_raw = RawArray(x, create_info(ch_names=[ch_name + "-tmp" for ch_name in DEEP_1010_CHS_LISTING], 
                                      sfreq=raw.info["sfreq"], 
                                      ch_types=DEEP_1010_TYPES))
    orig_ch_names = raw.ch_names

    # hacky way to add and replace channels to raw
    raw.add_channels([new_raw], force_update_info=True)
    raw.drop_channels(ch_names=orig_ch_names)
    raw.rename_channels({ch_name + "-tmp": ch_name for ch_name in DEEP_1010_CHS_LISTING})
    
    # not sure if this is completely necessary
    del new_raw
    del x

    return raw
    

def to_1020(raw):
    EEG_20_div = [
                'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'T5', 'P3', 'PZ', 'P4', 'T6',
                'O1', 'O2'
    ]

    _inds_20_div = [DEEP_1010_CHS_LISTING.index(ch) for ch in EEG_20_div]
    print('inds_20_div', _inds_20_div)
    _inds_20_div.append(SCALE_IND)

    ch_names_deep1010 = raw.ch_names
    print('ch_names_deep1010', ch_names_deep1010)
    print('dropping', [ch for i, ch in enumerate(ch_names_deep1010) if i not in _inds_20_div])
    raw.drop_channels([ch for i, ch in enumerate(ch_names_deep1010) if i not in _inds_20_div])
    print('left', raw.ch_names)
    return raw

def violates_nyquist(raw):
    lowpass = raw.info.get('lowpass', None)
    new_sfreq = 256.
    raw_sfreq = raw.info['sfreq']
    return lowpass is not None and (new_sfreq < 2 * lowpass) and new_sfreq != raw_sfreq

def interpolate_nearest(raw, sfreq=256.0):
    if not raw.preload:
        raw.load_data()
    
    x = raw._data

    old_sfreq = raw.info['sfreq']
    
    resampled_data = torch.nn.functional.interpolate(torch.tensor(x).unsqueeze(0), 
                                                     scale_factor=sfreq/old_sfreq, 
                                                     mode="nearest").squeeze(0).numpy()

    lowpass = raw.info.get("lowpass")
    with raw.info._unlock():
        raw.info["sfreq"] = sfreq
        raw.info["lowpass"] = min(lowpass, sfreq / 2.0)

    raw._data = resampled_data
    raw._last_samps = np.array([resampled_data.shape[1] - 1])
    return raw

if __name__ == "__main__":
    import pickle

    from braindecode.datasets import BaseConcatDataset
    from braindecode.preprocessing import Preprocessor, preprocess

    with open("/home/roraa/repos/EEGatScale/notebooks/tuh_ds_first10.pkl", "rb") as f:
        ds = pickle.load(f)

    ds = BaseConcatDataset(ds)

    preprocessors = [
        Preprocessor(to_deep1010, apply_on_array=False),
        Preprocessor(to_1020, apply_on_array=False)
        ]
    
    preprocess(ds, preprocessors)

    print(f"{ds.datasets[0].raw.get_data().shape=}")
