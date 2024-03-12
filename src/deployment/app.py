# app.py
import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.getcwd() + '/')
print(sys.path)
from src.data.utils.eeg import get_raw
from src.data.processing import load_data_dict, get_data
from src.data.conf.eeg_annotations import braincapture_annotations
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def main():
    st.title('Demonstration of EEG data pipeline')
    st.write("""
             This is a simple app for visualising and analysing EEG data. Start by uploading your .EDF file.
             """)
    
    edf_file_buffer = st.file_uploader('Upload .EDF file', type='edf')

    if edf_file_buffer is not None:
        raw = get_raw(os.path.join(os.getcwd() + '/data/v4.0/S001', edf_file_buffer.name))
        st.pyplot(raw.plot(n_channels=32, scalings='auto', title='BrainCapture EEG data'))

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
