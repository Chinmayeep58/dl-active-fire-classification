import streamlit as st
from global_styles import load_css

st.set_page_config(
    page_title="Satellite Labeling Tool",
    layout="wide"
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

load_css(st.session_state.theme)

home = st.Page(
    "0_Home.py",
    title="Home"
)

extraction_e = st.Page(
    "1_ESRI_Extraction.py",
    title="ESRI Extraction"
)

extraction_s = st.Page(
    "1_Sent_Extraction.py",
    title="Sentinel-2 Extraction"
)

labeling = st.Page(
    "2_Labeling.py",
    title="Model/Dataset loading and Labeling"
)

training = st.Page(
    "4_Feedback_Training.py",
    title="Feedback Training"
)

plotting = st.Page(
    "3_Model_Prediction.py",
    title="Model Prediction and Map Plotting"
)

pg = st.navigation([
    home,
    extraction_e,
    extraction_s,
    labeling,
    plotting,
    training
])

pg.run()
