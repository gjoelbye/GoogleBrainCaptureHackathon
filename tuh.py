import os
import numpy as np
import pandas as pd
import pyedflib

class TUH_data:
    def __init__(self, path):
        self.path = path
        self.subjects = {}
        self._process_data()

    def _process_data(self):
        for dirpath, dirnames, filenames in os.walk(self.path):
            for filename in [f for f in filenames if f.endswith(".edf") and "annotated" not in f]:
                session_path_split = os.path.split(dirpath)
                patient_path_split = os.path.split(session_path_split[0])
                subject_id = patient_path_split[1]
                session_id = session_path_split[1]

                if subject_id not in self.subjects:
                    self.subjects[subject_id] = {}

                if session_id not in self.subjects[subject_id]:
                    self.subjects[subject_id][session_id] = {"X": [], "y": []}

                edf_path = os.path.join(dirpath, filename)
                csv_path = os.path.join(dirpath, os.path.splitext(filename)[0] + '.csv')
                self.subjects[subject_id][session_id]["X"].append(edf_path)
                self.subjects[subject_id][session_id]["y"].append(csv_path)

    def write_annotations_to_edf(self, edf_path, annotations):
        # Read EDF file to get necessary information
        f = pyedflib.EdfReader(edf_path)
        num_signals = f.signals_in_file
        signal_labels = f.getSignalLabels()
        sample_frequency = f.getSampleFrequencies()[0]  # Assuming all signals have the same frequency

        # Create mapping from labels to integers
        def label_to_int(label):
            if label == 'musc':
                return 0
            elif label == 'other_label1':
                return 1
            else:
                return -1  # Return -1 if label not found

        # Create EDF writer
        new_edf_path = os.path.splitext(edf_path)[0] + '_annotated.edf'
        writer = pyedflib.EdfWriter(new_edf_path, n_channels=num_signals, file_type=pyedflib.FILETYPE_EDFPLUS)

        # Write signals to new EDF file
        for i in range(num_signals):
            signal_info = f.getSignalHeader(i)
            writer.setSignalHeader(i, signal_info)

        # Write annotations to annotations channel (last channel)
        annotations_channel = np.zeros(f.getNSamples()[0])
        # annotations_channel = ["" for x in range(f.getNSamples()[0])]
        for index, annotation in annotations.iterrows():
        # for annotation in annotations:
            start_sample = int(annotation['start_time'] * sample_frequency)
            end_sample = int(annotation['stop_time'] * sample_frequency)
            label = annotation['label']
            for i in range(start_sample, end_sample):
                annotations_channel[i] = label_to_int(label)

        writer.writePhysicalSamples(annotations_channel)
        
        # Close writer and reader
        writer.close()
        f.close()

    def preprocess(self):
        for subject_id, sessions in self.subjects.items():
            for session_id, data in sessions.items():
                X = []
                y = []
                for edf_path, csv_path in zip(data["X"], data["y"]):
                    X.append(self._load_edf(edf_path))
                    y.append(self._load_csv(csv_path))
                # TODO: Preprocess X and y
        return X, y

    def _load_edf(self, edf_path):
        # Load EDF file
        f = pyedflib.EdfReader(edf_path)
        # Read signals
        signals = [f.readSignal(i) for i in range(f.signals_in_file)]
        f.close()
        return signals

    def _load_csv(self, csv_path):
        return pd.read_csv(csv_path, skiprows=6, on_bad_lines='warn')

# Example usage:
dataset = TUH_data("data/TUH_data_sample")
X, y = dataset.preprocess()
dataset.write_annotations_to_edf("data/TUH_data_sample/010/00001006/s001_2003_04_28/00001006_s001_t001.edf", y[0])
