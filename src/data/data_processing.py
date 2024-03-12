import os
import mne
import numpy as np
from tqdm import tqdm

from data.utils.eeg import get_raw

def load_data_dict(data_folder_path: str, channel_picks: list, channel_order: list, annotation_dict: dict, tmin: float = -0.5, tlen: float = 6, labels: bool = False):
    """Loads the data from the data folder.
    Parameters
    ----------
    data_folder_path : str
        The path to the data folder.
    channel_config : list
        The configuration of the channels.
    tmin : float
        The start time.
    tlen : float
        The duration of an epoch.
    labels : bool
        Whether to include labels.
    Returns
    -------
    data_dict : dict
        The data dictionary.
    """
    data_dict = {}

    for subject in tqdm(os.listdir(data_folder_path)):
        data_dict[subject] = {}

        for session in os.listdir(data_folder_path + subject):
            session_name = session.split('.')[0]
            data_dict[subject][session_name] = {}
            
            edf_file_path = data_folder_path + subject + '/' + session
            raw = get_raw(edf_file_path, channel_picks, channel_order, filter=True)

            print(raw.annotations.description)

            if labels:
                # TODO: remove try-except, was added to handle TUAR data
                try:
                    events = mne.events_from_annotations(raw, event_id=annotation_dict, verbose=False)
                except:
                    print(f'No annotations in {subject} {session_name}')
                    data_dict[subject].pop(session_name)
                    continue

                tmax = tmin + tlen
                epochs = mne.Epochs(raw, events=events[0], tmin=tmin, tmax=tmax, event_repeated='merge', verbose='warning')

                y = epochs.events[:, 2]

                data_dict[subject][session_name]['y'] = epochs.events[:, 2]
            else:
                epochs = mne.make_fixed_length_epochs(raw, duration=tlen, preload=True, verbose=False)

            data_dict[subject][session_name]['X'] = epochs.get_data()
    
    return data_dict


def get_data(data_dict, subject_list = None):
    """Returns the data and labels.
    Parameters
    ----------
    data_dict : dict
        The data dictionary.
    subject_list : list
        The list of subjects.
    Returns
    -------
    X : np.array
        The data.
    y : np.array
        The labels.
    """
    if subject_list is None:
        subject_list = list(data_dict.keys())

    X = [data_dict[subject][session]['X'] for subject in subject_list for session in data_dict[subject].keys()]
    X = np.concatenate(X)

    if 'y' in data_dict[subject_list[0]][list(data_dict[subject_list[0]].keys())[0]]:
        y = [data_dict[subject][session]['y'] for subject in subject_list for session in data_dict[subject].keys()]
        y = np.concatenate(y)
        return X, y

    return X