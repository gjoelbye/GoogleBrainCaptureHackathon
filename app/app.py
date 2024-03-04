# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import mne

# Hackathon example app for file upload and visualization (edf file with EEG data)
def main():
    st.title('Hackathon Example App')
    st.write('This is a simple app to demonstrate .EDF file upload and visualization.')
        