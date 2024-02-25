# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import os
import braindecode
from tqdm import tqdm
import configparser
from dataLoadTest import BhutanDataSet
from dataLoadTest import load_config

# Load the configuration file
config = load_config()

# Load the Bhutan EEG dataset
# Load the Bhutan EEG dataset
bhutan = BhutanDataSet('/Users/benjaminfazal/Downloads/Bhutan Data/test_file.edf', subject=int(config['Bhutan']['subject']), 
                        session=int(config['Bhutan']['session']), event=config['Bhutan']['event'], 
                        event_id=int(config['Bhutan']['event_id']), tmin=int(config['Bhutan']['tmin']), 
                        tmax=int(config['Bhutan']['tmax']), baseline=None, 
                        filter=(float(config['Bhutan']['filter1']), float(config['Bhutan']['filter2'])))

# investigate x and y
print(bhutan.X.shape)

# visualize the data
# Plot the data
plt.figure(figsize=(7, 3))
plt.plot(bhutan.X[0, 0])
plt.xlabel('Time (samples)')
plt.ylabel('Voltage (uV)')
plt.title('EEG data')
plt.show()

# visualize annotations
raw = bhutan.raw
annotations = raw.annotations

mapping = {i: e for i, e in enumerate(annotations.description)}
events, event_id = mne.events_from_annotations(raw)

# Plot the annotations
fig, ax = plt.subplots(1, 1, figsize=(15, 3))
mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, event_id=event_id, axes=ax)
ax.set(title='Annotations')
plt.show()






