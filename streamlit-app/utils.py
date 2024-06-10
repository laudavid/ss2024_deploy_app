import os
import joblib
import streamlit as st

from streamlit_gsheets import GSheetsConnection


@st.cache_data()
def load_artifact(file):
    path_artifact = r"saved_models/"
    artifact = joblib.load(os.path.join(path_artifact, file))
    return artifact

