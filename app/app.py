# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import mne
import matplotlib.pyplot as plt
import os

# from knn import 

# # Hackathon example app for file upload and visualization (edf file with EEG data)

# def plot_knn():
#     f

def main():
    st.title('Hackathon Example App')
    st.write('This is a simple app to demonstrate .EDF file upload and visualization using MNE.')
    st.write('Current working directory: ', os.getcwd())

    uploaded_file = st.file_uploader('Upload an .EDF file', type='edf')

    if uploaded_file is not None:
        st.write('File uploaded successfully!')
        st.write('Loading file...')
        st.write(f'File name: {uploaded_file.name}')
        st.write(f'File type: {uploaded_file.type}')
        bytes_data = uploaded_file.read()
        raw = mne.io.read_raw_edf(bytes_data)
        st.write('File loaded successfully!')
        st.write('Displaying raw EEG data...')
        # plot raw
        raw.plot()
        st.pyplot()
        st.write('Displaying raw EEG data annotations...')
        # plot annotations
        raw.plot(events=raw.annotations, start=0, duration=10)

if __name__ == '__main__':
    main()