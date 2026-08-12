import streamlit as st

def load_css(theme="dark"):

    if theme == "dark":
        colors = {
            "bg": "#0D1B2A",
            "sidebar": "#111F2E",
            "card": "#111F2E",
            "text": "#F0EDE8",
            "secondary": "#D1D5DB",
            "border": "#1E3A2F",
            "hover": "#1A2A3A",
            "metric": "#111F2E",
            "header": "#0D1B2A",
        }
    else:
        colors = {
            "bg": "#F8FAFC",
            "sidebar": "#FFFFFF",
            "card": "#FFFFFF",
            "text": "#111827",
            "secondary": "#374151",
            "border": "#D1D5DB",
            "hover": "#F3F4F6",
            "metric": "#FFFFFF",
            "header": "#F8FAFC",
        }

    st.markdown(
        f"""
        <style>

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {colors["text"]};
        }}

        .stApp {{
            background-color: {colors["bg"]};
            color: {colors["text"]};
        }}

        /* Header */
        header[data-testid="stAppHeader"]{{
            background-color: {colors["header"]};
        }}
        
        .block-container{{
            padding-top: 5rem;
            max-width: 1100px;
        }}

        /* Headings */
        h1, h2, h3, h4, h5, h6 {{
            color: {colors["text"]} !important;
        }}

        p, span, div, label {{
            color: {colors["secondary"]};
        }}

        .stMarkdown {{
            color: {colors["secondary"]};
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {colors["sidebar"]};
            border-right: 1px solid {colors["border"]};
        }}

        section[data-testid="stSidebar"] * {{
            color: {colors["text"]} !important;
        }}

        /* Navigation */
        [data-testid="stSidebarNav"] li a {{
            color: {colors["secondary"]} !important;
            border-radius: 8px;
            transition: all 0.2s ease;
        }}

        [data-testid="stSidebarNav"] li a:hover {{
            background-color: {colors["hover"]};
            color: {colors["text"]} !important;
        }}

        [data-testid="stSidebarNav"] li a[aria-current="page"] {{
            background-color: rgba(232,98,42,0.12) !important;
            border-left: 3px solid #E8622A;
            color: {colors["text"]} !important;
            font-weight: 600;
        }}

        /* Inputs */
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {{
            background-color: {colors["card"]} !important;
            color: {colors["text"]} !important;
            border: 1px solid {colors["border"]} !important;
        }}

        /* Selectbox */
        .stSelectbox div[data-baseweb="select"] {{
            background-color: {colors["card"]};
            color: {colors["text"]};
        }}

        /* File uploader */
        [data-testid="stFileUploader"] {{
            background-color: {colors["card"]};
            border-radius: 8px;
            border: 1px solid {colors["border"]};
        }}

        /* Buttons */
        .stButton > button {{
            background-color: #E8622A;
            color: white !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }}

        .stButton > button:hover {{
            background-color: #d4541f;
            color: white !important;
        }}

        /* Metrics */
        [data-testid="stMetric"] {{
            background-color: {colors["metric"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            padding: 12px;
        }}

        [data-testid="stMetricValue"] {{
            color: {colors["text"]} !important;
        }}

        [data-testid="stMetricLabel"] {{
            color: {colors["secondary"]} !important;
        }}

        /* Dataframes */
        .stDataFrame,
        .stTable {{
            background-color: {colors["card"]};
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            color: {colors["secondary"]};
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: #E8622A;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )