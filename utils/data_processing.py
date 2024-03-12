import os
import mne
import numpy as np
from tqdm import tqdm

from utils.eeg import get_raw

def anonymize_folder_naming(data_folder_path):
    """Anonymizes the folder naming.
    Parameters
    ----------
    data_folder_path : str
        The path to the data folder.
    """
    subject_folders = os.listdir(data_folder_path)

    for i, subject_folder in enumerate(subject_folders):
        subject_folder_path = data_folder_path + subject_folder + '/'
        session_files = os.listdir(subject_folder_path)

        for j, session_file in enumerate(session_files):
            session_file_path = subject_folder_path + session_file
            new_session_file_path = subject_folder_path + f'S{str(i+1).zfill(3)}R{str(j+1).zfill(2)}.edf'

            if not os.path.exists(new_session_file_path):
                os.rename(session_file_path, new_session_file_path)

        new_subject_folder_path = data_folder_path + f'S{str(i+1).zfill(3)}/'

        if not os.path.exists(new_subject_folder_path):
            os.rename(subject_folder_path, new_subject_folder_path)

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

            #print(raw.annotations.description)

            if labels:
                # TODO: remove try-except, was added to handle TUAR data
                try:
                    events = mne.events_from_annotations(raw, event_id=annotation_dict, verbose=False)
                except:
                    print(f'No annotations in {subject} {session_name}')
                    data_dict[subject].pop(session_name)
                    continue

                tmax = tmin + tlen
                epochs = mne.Epochs(raw, events=events[0], tmin=tmin, tmax=tmax, event_repeated='merge', verbose=False, baseline=(0, 0))

                y = epochs.events[:, 2]

                # TODO: remove this hack
                # if 4 in y or 5 in y or 6 in y or 7 in y:
                #     print(f'{subject} {session_name} has fishy annotations, skipping...')
                #     data_dict[subject].pop(session_name)
                #     continue

                data_dict[subject][session_name]['y'] = epochs.events[:, 2]
            else:
                epochs = mne.make_fixed_length_epochs(raw, duration=tlen, preload=True, verbose=False)

            data_dict[subject][session_name]['X'] = epochs.get_data(verbose=False)
                       
    return data_dict

# TODO: better name for this function
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


def prepare_for_hackathon(source_data_folder_path: str, hacktahon_data_folder_path: list, channel_picks: list, channel_order: list):
    """Prepares the data for the hackathon.
    Parameters
    ----------
    source_data_folder_path : str
        The path to the source data folder.
    hacktahon_data_folder_path : list
        The path to the hackathon data folder.
    channel_picks : list
        The configuration of the channels.
    channel_order : list
        The order of the channels.
    """
    for subject in tqdm(os.listdir(source_data_folder_path)):
        for session in os.listdir(source_data_folder_path + subject):
            edf_file_path = source_data_folder_path + subject + '/' + session
            raw = get_raw(edf_file_path, channel_picks, channel_order, filter=True)

            output_path = hacktahon_data_folder_path + subject

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            raw.export(output_path + '/' + session, fmt='edf', overwrite=True)