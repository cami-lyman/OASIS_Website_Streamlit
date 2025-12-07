import streamlit as st
import pandas as pd
import math
from pathlib import Path
import os
from PIL import Image
import time
from typing import List
import numpy as np

os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"

# Try to import plotting libraries (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# ------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------
st.set_page_config(
    page_title='Examining the Relationship between Brain Volume and Dementia Diagnoses',
    page_icon=':brain:',
    layout='wide',
)

# ------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------
@st.cache_data
def get_data(filename=None):
    if filename is None:
        filename = Path(__file__).parent / "data/final_data_oasis.csv"

    try:
        df = pd.read_csv(filename)
        return df
    except Exception:
        return None

# ------------------------------------------------------------------------
# MRI SLICE VIEWER HELPERS
# ------------------------------------------------------------------------
SLICE_DIR = "oasis/mri_files"

@st.cache_data
def list_slices(slice_dir: str) -> List[str]:
    try:
        files = [f for f in os.listdir(slice_dir) if f.lower().endswith(".png")]
        return sorted(files)[::-1]
    except Exception:
        return []

@st.cache_data
def load_image(path: str):
    try:
        return Image.open(path)
    except Exception:
        return None

if "slice_index" not in st.session_state:
    st.session_state.slice_index = 0

if "play" not in st.session_state:
    st.session_state.play = False

def update_slice():
    st.session_state.slice_index = st.session_state.slice_slider

# ------------------------------------------------------------------------
# OVERVIEW PAGE
# ------------------------------------------------------------------------
def render_overview():
    st.title('Examining the Relationship between Brain Volume and Dementia Diagnoses :)')
    st.write('An analysis of data provided by the OASIS project.')

    text_col, image_col = st.columns([1, 1])

    with text_col:
        st.subheader("About")
        st.write("""
        This project examines the relationship between brain volume measurements 
        and dementia diagnoses using data from the OASIS project.
        """)

    with image_col:
        st.subheader("MRI Slice Viewer")
        slice_files = list_slices(SLICE_DIR)
        if not slice_files:
            st.warning('No MRI slices found.')
            return

        max_idx = len(slice_files) - 1

        # Controls
        c1, c2, c3 = st.columns([1, 1, 1])

        if c1.button("◀ Prev"):
            st.session_state.slice_index = max(0, st.session_state.slice_index - 1)
            st.session_state.play = False

        if c2.button("⏯ Play/Pause"):
            st.session_state.play = not st.session_state.play

        if c3.button("Next ▶"):
            st.session_state.slice_index = min(max_idx, st.session_state.slice_index + 1)
            st.session_state.play = False

        # Slider
        st.slider(
            "Slice",
            0,
            max_idx,
            value=st.session_state.slice_index,
            key="slice_slider",
            on_change=update_slice,
        )

        # Image
        img_path = os.path.join(SLICE_DIR, slice_files[st.session_state.slice_index])
        img = load_image(img_path)
        st.image(img, caption=f"Slice {st.session_state.slice_index}", width="stretch")

    if st.session_state.play:
        time.sleep(0.2)
        st.session_state.slice_index = (st.session_state.slice_index + 1) % (max_idx + 1)
        st.rerun()

# ------------------------------------------------------------------------
# OASIS PAGE
# ------------------------------------------------------------------------
def render_oasis():
    st.header('OASIS', divider='blue')
    st.write('Explain the OASIS project and dataset here.')

# ------------------------------------------------------------------------
# CODE PAGE
# ------------------------------------------------------------------------
def render_code():
    st.header('Code', divider='blue')
    try:
        src = Path(__file__).read_text()
        st.code(src, language='python')
    except:
        st.warning("Unable to load source file.")

# ------------------------------------------------------------------------
# DATA & GRAPHS
# ------------------------------------------------------------------------
def render_data_and_graphs():
    st.header('Data & Graphs', divider='blue')

    df_local = get_data()
    if df_local is None:
        st.warning("Dataset not found.")
        return

    st.subheader("Data preview")
    st.dataframe(df_local.iloc[:, 1:], height=210)

    volume_methods = ['nWBV', 'nWBV_brain_extraction', 'nWBV_deep_atropos']
    method_labels = {
        'nWBV': 'nWBV (Original)',
        'nWBV_brain_extraction': 'nWBV (Brain Extraction)',
        'nWBV_deep_atropos': 'nWBV (Deep Atropos)'
    }

    available_methods = [m for m in volume_methods if m in df_local.columns]
    if not available_methods:
        st.warning("Required brain volume columns missing.")
        return

    method_colors = {
        'nWBV': 'tab:orange',
        'nWBV_brain_extraction': 'tab:green',
        'nWBV_deep_atropos': 'tab:blue'
    }

    # ------------------------------
    # HISTOGRAMS
    # ------------------------------
    st.subheader('Distribution of Brain Volume — Histograms')
    fig, axes = plt.subplots(1, len(available_methods), figsize=(9 * len(available_methods), 6))

    if len(available_methods) == 1:
        axes = [axes]

    for idx, method in enumerate(available_methods):
        vals = df_local[method].dropna()
        sns.histplot(vals, bins=20, color=method_colors[method], ax=axes[idx])
        axes[idx].set_title(method_labels[method])
        axes[idx].set_xlabel("Brain Volume")
        axes[idx].set_ylabel("Count")

    st.pyplot(fig)

    # ------------------------------
    # BOXPLOTS
    # ------------------------------
    if "CDR" in df_local.columns:
        st.subheader('Brain Volume by CDR — Boxplots')
        fig, axes = plt.subplots(1, len(available_methods), figsize=(9 * len(available_methods), 6))

        if len(available_methods) == 1:
            axes = [axes]

        for idx, method in enumerate(available_methods):
            sns.boxplot(x="CDR", y=method, data=df_local, ax=axes[idx], color=method_colors[method])
            axes[idx].set_title(method_labels[method])
            axes[idx].set_xlabel("CDR")
            axes[idx].set_ylabel("Brain Volume")

        st.pyplot(fig)

    # ------------------------------
    # SCATTERPLOTS (with legend)
    # ------------------------------
    st.subheader("Brain Volume vs Age — Scatterplots")

    age_cols = ['AGE', 'Age', 'age']
    sex_cols = ['M/F', 'SEX', 'Sex', 'sex', 'Gender', 'gender']

    age_col = next((c for c in age_cols if c in df_local.columns), None)
    sex_col = next((c for c in sex_cols if c in df_local.columns), None)

    if age_col is None or sex_col is None:
        st.warning("Dataset missing age or sex column.")
        return

    fig, axes = plt.subplots(1, len(available_methods), figsize=(9 * len(available_methods), 6))
    if len(available_methods) == 1:
        axes = [axes]

    # Legend handles
    female_handle = plt.Line2D([], [], marker='o', color='red', linestyle='None', label='Female')
    male_handle = plt.Line2D([], [], marker='o', color='blue', linestyle='None', label='Male')

    def _sex_color(v):
        s = str(v).strip().lower()
        if s in ("f", "female"): return "red"
        if s in ("m", "male"): return "blue"
        return "lightgray"

    for idx, method in enumerate(available_methods):
        df_plot = df_local[[age_col, method, sex_col]].dropna()
        colors = df_plot[sex_col].map(_sex_color)

        axes[idx].scatter(
            df_plot[age_col],
            df_plot[method],
            c=colors,
            alpha=0.8,
            edgecolor='k'
        )

        axes[idx].set_title(method_labels[method])
        axes[idx].set_xlabel(age_col)
        axes[idx].set_ylabel("Brain Volume")

        axes[idx].legend(handles=[female_handle, male_handle], fontsize=12)

    st.pyplot(fig)

# ------------------------------------------------------------------------
# CONCLUSIONS
# ------------------------------------------------------------------------
def render_conclusions():
    st.header("Conclusions", divider="blue")
    st.write("Summarize findings here.")

# ------------------------------------------------------------------------
# REFERENCES
# ------------------------------------------------------------------------
def render_references():
    st.header("References", divider="blue")
    st.write("List references here.")

# ------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------
st.sidebar.header("Navigation")

PAGES = [
    "Overview",
    "OASIS",
    "Code",
    "Data & Graphs",
    "Conclusions",
    "References"
]

if "page" not in st.session_state:
    st.session_state.page = "Overview"

page = st.sidebar.radio("Navigation", PAGES, index=PAGES.index(st.session_state.page))
st.session_state.page = page

# ------------------------------------------------------------------------
# CUSTOM SIDEBAR BUTTON STYLING (restored)
# ------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <style>

    /* Hide default radio circles */
    section[data-testid="stSidebar"] input[type="radio"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] label > div:first-child {
        display: none !important;
    }

    /* Sidebar layout adjustment */
    section[data-testid="stSidebar"] .stRadio {
        margin-top: -0.5rem !important;
    }

    /* Unselected button style */
    section[data-testid="stSidebar"] .stRadio label {
        display: block !important;
        padding: 10px 14px !important;
        margin: 4px 0 !important;
        border-radius: 10px !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;

        background-color: #f4f4f4 !important;
        border: 1.5px solid #cccccc !important;
        font-weight: 500 !important;
        color: #333333 !important;
    }

    /* Hover effect */
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: #e8e8e8 !important;
    }

    /* Selected button style */
    section[data-testid="stSidebar"] .stRadio label:has(input[type="radio"]:checked) {
        background-color: #4a90e2 !important;
        color: white !important;
        border: 1.5px solid #2d6fba !important;
        font-weight: 600 !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.25) !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Render selected page
if page == "Overview":
    render_overview()
elif page == "OASIS":
    render_oasis()
elif page == "Code":
    render_code()
elif page == "Data & Graphs":
    render_data_and_graphs()
elif page == "Conclusions":
    render_conclusions()
elif page == "References":
    render_references()
