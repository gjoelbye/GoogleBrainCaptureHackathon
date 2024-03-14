# app.py
import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.getcwd() + '/')
from src.data.utils.eeg import get_raw
from src.data.processing import load_data_dict, get_data
from src.data.conf.eeg_annotations import braincapture_annotations
from src.data.conf.eeg_channel_picks import hackathon
from src.data.conf.eeg_channel_order import standard_19_channel
from src.data.conf.eeg_annotations import braincapture_annotations, tuh_eeg_artefact_annotations
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tempfile

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

def main():
    st.title('Demonstration of EEG data pipeline')
    st.write("""
             This is a simple app for visualising and analysing EEG data. Start by uploading your .EDF files you want to analyse.
             """)
    
    edf_file_buffers = st.file_uploader('Upload .EDF files', type='edf', accept_multiple_files=True)
    
    if edf_file_buffers:
        # for edf_file_buffer in edf_file_buffers:
        data_folder, file_paths = get_file_paths(edf_file_buffers)
        
        
        if st.button("Process data"):
            st.write("Data processing initiated")
            st.write(f"your file paths: {file_paths}")
            for file_path in file_paths:
                raw = get_raw(file_path)
                st.pyplot(raw.plot(n_channels=32, scalings='auto', title='BrainCapture EEG data'))

            data_dict = load_data_dict(data_folder_path=data_folder, annotation_dict=braincapture_annotations, tmin=-0.5, tlen=6, labels=True)
            type(data_dict)
            # raw = get_raw(file_path)
            # raws.append(raw)
            # st.pyplot(raw.plot(n_channels=32, scalings='auto', title='BrainCapture EEG data'))
        

    # if edf_file_buffer is not None:
    #     # raw = get_raw(os.path.join(os.getcwd() + '/data/v4.0/S001', edf_file_buffer.name))
    #     raw = get_raw(edf_file_buffer)
    #     st.pyplot(raw.plot(n_channels=32, scalings='auto', title='BrainCapture EEG data'))

    # if img_file_buffer is not None:
    #     image = Image.open(img_file_buffer)
    #     img_array = np.array(image)

    
    #     st.image(
    #         image,
    #         caption=f"You amazing image has shape {img_array.shape[0:2]}",
    #         use_column_width=True,
    # )
    
    # plot_decision_boundary(X_train_pca, y_train, targets, knn)
    # visualize_PCA(X_train_pca, y_train, targets, split='Training')
    # visualize_PCA(X_test_pca, y_test, targets, split='Testing')
main()
