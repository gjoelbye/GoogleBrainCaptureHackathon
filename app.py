# Note: The GCP bucket with model weights is mounted at: /bucket/ on the file system
#       our weights are in /bucket/big-bucketz/
#       /bucket/big-bucketz/best_classification_model.pt and /bucket/big-bucketz/best_binary_model.pt
# app.py
import streamlit as st
import sys
import os
sys.path.append(os.getcwd() + '/')
from src.data.processing import load_data_dict, get_data
from src.data.conf.eeg_annotations import braincapture_annotations
from src.data.conf.eeg_annotations import braincapture_annotations
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import torch
from tqdm import tqdm
from copy import deepcopy
from model.model import BendrEncoder
from model.classifiers import create_binary_model, create_classification_model

max_length = lambda raw : int(raw.n_times / raw.info['sfreq']) 
DURATION = 60
device = 'cpu'

def generate_latent_representations(data, encoder, batch_size=5, device='cpu'):
    """ Generate latent representations for the given data using the given encoder.
    Args:
        data (np.ndarray): The data to be encoded.
        encoder (nn.Module): The encoder to be used.
        batch_size (int): The batch size to be used.
    Returns:
        np.ndarray: The latent representations of the given data.
    """
    data = data.to(device)

    latent_size = (1536, 4) # do not change this 
    latent = np.empty((data.shape[0], *latent_size))


    for i in tqdm(range(0, data.shape[0], batch_size)):
        latent[i:i+batch_size] = encoder(data[i:i+batch_size]).cpu().detach().numpy()

    return latent.reshape((latent.shape[0], -1))

def load_model(device='cpu'):
    """Loading BendrEncoder model
    Args:
        device (str): The device to be used.
    Returns:
        BendrEncoder (nn.Module): The model
    """

    # Initialize the model
    encoder = BendrEncoder()

    # Load the pretrained model
    encoder.load_state_dict(deepcopy(torch.load("encoder.pt", map_location=device)))
    encoder = encoder.to(device)

    return encoder

def make_dir(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass

def get_file_paths(edf_file_buffers):
    """
    input: edf_file_buffers: list of files uploaded by user

    output: paths: paths to the files
    """
    paths = []
    # make tempoary directory to store the files
    temp_dir = tempfile.mkdtemp()
    print(temp_dir)
    for edf_file_buffer in edf_file_buffers:
        folder_name = os.path.join(temp_dir, edf_file_buffer.name[:4])
        make_dir(folder_name)
        # make tempoary file
        path = os.path.join(folder_name , edf_file_buffer.name)
        # write bytesIO object to file
        with open(path, 'wb') as f:
            f.write(edf_file_buffer.getvalue())

        paths.append(path)

    return temp_dir + '/', paths

def plot_clusters(components, labels):
    """
    input: 
        components: 2D array of the principal components
        labels: labels of the clusters
    
    output: None"""

    # Plot clusters
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for cluster_label in unique_labels:
        ax.scatter(components[labels == cluster_label, 0], components[labels == cluster_label, 1], label=f'Cluster {cluster_label}')

    ax.set_title('Clusters using PCA')
    ax.set_xlabel('Principal Component 0')
    ax.set_ylabel('Principal Component 1')
    ax.legend()

    st.pyplot(fig)

def main():
    st.title('Demonstration of EEG data pipeline')
    st.write("""
             This is a simple app for visualising and analysing EEG data. Start by uploading the .EDF files you want to analyse.
             """)
    
    # 1: Upload EDF files
    edf_file_buffers = st.file_uploader('Upload .EDF files', type='edf', accept_multiple_files=True)

    if edf_file_buffers:
        data_folder, file_paths = get_file_paths(edf_file_buffers)
        
        if st.button("Process data"):
            st.write("Data processing initiated")
          
            # 2: Chop the .edf data into 5 second windows
            data_dict = load_data_dict(data_folder_path=data_folder, annotation_dict=braincapture_annotations, tlen=5, labels=False)
            all_subjects = list(data_dict.keys())
            X = get_data(data_dict, all_subjects)

            # 3: Do binary predictions
            binary_model = create_binary_model()
            binary_model.load_state_dict("/bucket/big-bucketz/best_binary_model.pt")
            
            # 4: Do multi class predictions
            class_model = create_classification_model()
            class_model.load_state_dict("/bucket/big-bucketz/best_classification_model.pt")



main()
