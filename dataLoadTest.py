# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import os
import braindecode
from tqdm import tqdm
import configparser

# Make the dataset class
class BhutanDataSet():
    """
    Defines the BhutanDataSet class. This class is used to load the Bhutan EEG dataset and preprocess it for
    later use in the BENDR feature representation generation. It is loaded as a raw .edf file and the preprocessed 
    according to Makoto's pipeline.

    """
    def __init__(self, path, subject, session, event, event_id, tmin, tmax, baseline, filter):
        """
        Initializes the BhutanDataSet class.

        Args:
            path (str): The path to the raw .edf file.
            subject (int): The subject number.
            session (int): The session number.
            event (str): The event to be extracted.
            event_id (int): The event id.
            tmin (float): The minimum time to be extracted.
            tmax (float): The maximum time to be extracted.
            baseline (tuple): The baseline to be used.
            filter (tuple): The filter to be used.

        """
        self.path = path
        self.subject = subject
        self.session = session
        self.event = event
        self.event_id = event_id
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.filter = filter

        print("Loading the Bhutan EEG dataset for subject {} and session {}...".format(self.subject, self.session))

        self.raw, self.epochs, self.X, self.y = self.call()

    def call(self):
        """
        Calls the BhutanDataSet class.

        Returns:
            raw (mne.io.edf.edf.RawEDF): The raw .edf file.
            epochs (mne.epochs): The preprocessed epochs.
            X (np.array): The data.
            y (np.array): The labels.

        """
        raw = self.load_data()
        # self.visualise_data(raw)
        epochs = self.preprocess_data(raw)
        X, y = self.extract_data(epochs)
        return raw, epochs, X, y

    def load_data(self):
        """
        Loads the raw .edf file.

        Returns:
            raw (mne.io.edf.edf.RawEDF): The raw .edf file.

        """
        raw = mne.io.read_raw_edf(self.path, infer_types=True, verbose=True)
        # Print the number of channels and the channel names
        print("The number of channels is: ", len(raw.info['ch_names']))
        print("The channel names are: ", raw.info['ch_names'])
        raw.drop_channels(['R1', 'R2', 'TIP', 'GROUND', 'REF']) # BAD PRACTICE - REMOVE
        return raw
    
    def visualise_data(self, raw):
        """
        Visualises the raw .edf file.

        Args:
            raw (mne.io.edf.edf.RawEDF): The raw .edf file.

        """
        raw.plot()
    
    def preprocess_data(self, raw):
        """
        Preprocesses the raw .edf file.

        Args:
            raw (mne.io.edf.edf.RawEDF): The raw .edf file.

        Returns:
            epochs (mne.epochs): The preprocessed epochs.

        """
        # Set the montage
        montage = mne.channels.make_standard_montage('standard_1020')
        # for montage in mne.channels.get_builtin_montages():
        #     try:
        #         raw.set_montage(montage, match_case=False)
        #         print('Set montage to', montage)
        #         break
        #     except Exception as e:
        #         print(e, 'for', montage)
        raw.set_montage(montage, match_case=False)
        # Load data into memory
        raw.load_data()
        # Filter the data
        raw.filter(self.filter[0], self.filter[1], fir_design='firwin')
        # Extract events
        events, event_id = mne.events_from_annotations(raw)
        # Extract epochs
        # epochs = mne.Epochs(raw, events, event_id, tmin=self.tmin, tmax=self.tmax, baseline=self.baseline, preload=True)
        epochs = mne.Epochs(raw, events, baseline=self.baseline, preload=True)
        return epochs
    
    def extract_data(self, epochs):
        """
        Extracts the data from the epochs.

        Args:
            epochs (mne.epochs): The preprocessed epochs.

        Returns:
            X (np.array): The data.
            y (np.array): The labels.

        """
        X = epochs.get_data()
        y = epochs.events[:, -1]
        return X, y
    
class MMIDBDataSet():
    """
    Defines the MMIDBDataSet class. This class is used to load the MMIDB EEG dataset and preprocess it for
    later use in the BENDR feature representation generation. It is loaded as a raw .edf file and the preprocessed 
    according to Makoto's pipeline.

    """
    def __init__(self, path, subject, session, event, event_id, tmin, tmax, baseline, filter):
        """
        Initializes the MMIDBDataSet class.

        Args:
            path (str): The path to the raw .edf file.
            subject (int): The subject number.
            session (int): The session number.
            event (str): The event to be extracted.
            event_id (int): The event id.
            tmin (float): The minimum time to be extracted.
            tmax (float): The maximum time to be extracted.
            baseline (tuple): The baseline to be used.
            filter (tuple): The filter to be used.

        """
        self.path = path
        self.subject = subject
        self.session = session
        self.event = event
        self.event_id = event_id
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.filter = filter

        print("Loading the MMIDB EEG dataset for subject {} and session {}...".format(self.subject, self.session))

        self.raw, self.epochs, self.X, self.y = self.call()

    def call(self):
        """
        Calls the MMIDBDataSet class.

        Returns:
            raw (mne.io.edf.edf.RawEDF): The raw .edf file.
            epochs (mne.epochs): The preprocessed epochs.
            X (np.array): The data.
            y (np.array): The labels.

        """
        raw = self.load_data()
        # self.visualise_data(raw)
        epochs = self.preprocess_data(raw)
        X, y = self.extract_data(epochs)
        return raw, epochs, X, y

    def load_data(self):
        """
        Loads the raw .edf file.

        Returns:
            raw (mne.io.edf.edf.RawEDF): The raw .edf file.

        """
        raw = mne.io.read_raw_edf(self.path, infer_types=True, verbose=True)
        # Print the number of channels and the channel names
        print("The number of channels is: ", len(raw.info['ch_names']))
        print("The channel names are: ", raw.info['ch_names'])

        # Rename channelse, remove all .'s and spaces
        raw.rename_channels(lambda x: x.strip('.').replace(' ', ''))
        return raw
    
    def visualise_data(self, raw):
        """
        Visualises the raw .edf file.

        Args:
            raw (mne.io.edf.edf.RawEDF): The raw .edf file.

        """
        raw.plot()

    def preprocess_data(self, raw):
        """
        Preprocesses the raw .edf file.

        Args:
            raw (mne.io.edf.edf.RawEDF): The raw .edf file.

        Returns:
            epochs (mne.epochs): The preprocessed epochs.

        """
        # Set the montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False)
        # Load data into memory
        raw.load_data()
        # Filter the data
        raw.filter(self.filter[0], self.filter[1], fir_design='firwin')
        # Extract events
        events, event_id = mne.events_from_annotations(raw)
        # Extract epochs
        epochs = mne.Epochs(raw, events, event_id, tmin=self.tmin, tmax=self.tmax, baseline=self.baseline, preload=True)
        return epochs
    
    def extract_data(self, epochs):
        """
        Extracts the data from the epochs.

        Args:
            epochs (mne.epochs): The preprocessed epochs.

        Returns:
            X (np.array): The data.
            y (np.array): The labels.

        """
        X = epochs.get_data()
        y = epochs.events[:, -1]
        return X, y
    

    
def load_config(path='config.ini'):
    """
    Loads the configuration file.

    Returns:
        config (configparser.ConfigParser): The configuration file.

    """
    config = configparser.ConfigParser()
    config.read(path)
    return config

if __name__ == "__main__":
    # Load the configuration file
    config = load_config()

    # Load the Bhutan EEG dataset
    bhutan = BhutanDataSet('data/Bhutan_Data/test_file.edf', subject=int(config['Bhutan']['subject']), 
                           session=int(config['Bhutan']['session']), event=config['Bhutan']['event'], 
                           event_id=int(config['Bhutan']['event_id']), tmin=int(config['Bhutan']['tmin']), 
                           tmax=int(config['Bhutan']['tmax']), baseline=None, 
                           filter=(float(config['Bhutan']['filter1']), float(config['Bhutan']['filter2'])))

    # investigate x and y
    print(bhutan.X.shape)

    # Load the MMIDB EEG dataset
    mmidb = MMIDBDataSet('data/MMIDB/S013R01.edf', subject=int(config['MMIDB']['subject']), 
                         session=int(config['MMIDB']['session']), event=config['MMIDB']['event'], 
                         event_id=int(config['MMIDB']['event_id']), tmin=int(config['MMIDB']['tmin']), 
                         tmax=int(config['MMIDB']['tmax']), baseline=None, 
                         filter=(float(config['MMIDB']['filter1']), float(config['MMIDB']['filter2'])))
    
    # investigate x and y
    print(mmidb.X.shape)