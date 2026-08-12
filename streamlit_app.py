# import streamlit as st

# st.set_page_config(
#     page_title="Satellite Labeling Tool",
#     layout="wide"
# )

# home=st.Page(
#     "0_Home.py",
#     title="Home"
# )

# extraction_e = st.Page(
#     "1_ESRI_Extraction.py",
#     title="ESRI Extraction"
# )

# extraction_s = st.Page(
#     "1_Sent_Extraction.py",
#     title="Sentinel-2 Extraction"
# )

# labeling = st.Page(
#     "2_Labeling.py",
#     title="Model/Dataset loading and Labeling"
# )

# labeling_wc = st.Page(
#     "2_Labeling_wc.py",
#     title="Labeling (choice)"
# )

# training = st.Page(
#     "3_Feedback_Training.py",
#     title="Feedback Training"
# )

# plotting=st.Page(
#     "101_Model_Prediction.py",
#     title="Model Prediction and Map Plotting"
# )

# pg = st.navigation([
#     home,
#     extraction_e,
#     extraction_s,
#     labeling,
#     plotting,
#     training
# ])

# pg.run()

import streamlit as st
from global_styles import load_css

st.set_page_config(
    page_title="Satellite Labeling Tool",
    layout="wide"
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# st.sidebar.markdown("### Appearance")

# dark_mode = st.sidebar.toggle(
#     "🌙 Dark Mode",
#     value=st.session_state.theme == "dark"
# )

# st.session_state.theme = (
#     "dark" if dark_mode else "light"
# )

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