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
    st.title('Examining the Relationship between Brain Volume and Dementia Diagnoses')
    st.write('An analysis of data provided by the OASIS project.')

    text_col, image_col = st.columns([1, 1])

    with text_col:
        st.subheader("About")
        st.write("""
        Dementia is a neurodegenerative disease which impacts millions of people around the world. Currently, the average time to diagnosis is 3.5 years [1]. This delay reduces the treatment options available, as many treatments that slow disease progression are only effective in the early stages. Brain MRIs may offer a way to improve early diagnosis and track disease progression, as some brain changes can be seen years before symptoms develop [2]. Neuroinflammation is a likely part of the pathogenesis of Alzheimer’s, and it can be seen on MRIs [3][4]. Additionally, loss of brain volume over time is a known feature of dementia.

        Our goal was to examine the relationship between brain volume and dementia status using data from the OASIS (Open Access Series of Imaging Studies) project. The OASIS project dataset we used contained three-dimensional MRI scan files as well as information about each patient’s dementia status as measured by clinical dementia rating (CDR) and demographic information such as age and gender. The MRI viewer on the right displays cross-sectional brain scans that allow you to explore the anatomy captured in this dataset. You can learn more about the OASIS project on the “OASIS” page.

        We derived the brain volume from the MRI files by slicing them into two-dimensional images, segmenting the image to calculate brain area, and adding up the brain area from each slice. We used two different deep learning models within ANTsPy to segment the brain areas: brain extraction and deep atropos. You can see more about our methods on the “Code” page. Our results can be found on the “Data & Graphs” and “Conclusions” pages.

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
        time.sleep(0.08)
        st.session_state.slice_index = (st.session_state.slice_index + 1) % (max_idx + 1)
        st.rerun()

# ------------------------------------------------------------------------
# OASIS PAGE
# ------------------------------------------------------------------------
def render_oasis():
    st.header("OASIS")

    st.subheader("About the OASIS Project")
    st.write("""
    In their words, *“The Open Access Series of Imaging Studies (OASIS) is a project aimed at 
    making neuroimaging data sets of the brain freely available to the scientific community. 
    By compiling and freely distributing neuroimaging data sets, we hope to facilitate future 
    discoveries in basic and clinical neuroscience,”* [5].

    The data in these datasets are gathered from the Knight ADRC and affiliated studies. 
    Participants include both men and women, and the dataset controls for handedness by 
    including only right-handed participants. Although handedness does not affect total 
    brain volume, it does influence hemispheric asymmetry patterns, which could matter for 
    studies exploring lateralized features of the brain [6].
    """)

    st.subheader("Dataset Used")
    st.write("""
    For our project, we used the **OASIS-1** dataset, which includes **416 participants** 
    aged **18 to 96**. Each participant contributed **three to four T1-weighted MRI scans** 
    taken during the same imaging session. Of the 416 participants, 100 were clinically diagnosed with dementia,
    and 20 non-demented participants were imaged on a subsequent visit within 90 days of the initial session to use as a reliability dataset.
    """)

    st.subheader("Available Participant Metadata")
    st.write("""
    In addition to the brain scans, sex, handedness, age, education level, socioeconomic status, MMSE, CDR, estimated total intracranial volume, normalized whole brain volume, and ATLAS scaling factor were recorded for each participant. CDR and MMSE are tools used to gauge the stage of dementia. We chose to focus on CDR. To obtain CDR, physicians conduct a semi-structured interview with the patient and a reliable informant such as a family member to assess six domains of cognitive and functional performance: Memory, Orientation, Judgment & Problem Solving, Community Affairs, Home & Hobbies, and Personal Care. A higher score means more severe dementia. The OASIS study used atlas scaling to account for differences in head size before calculating the normalized whole brain volume for each participant.

    """)

    st.subheader("MRI Data Used")
    st.write("""
    We used the **gain field–corrected, ATLAS-registered transverse brain scans** for all analyses. 
    ATLAS registration normalizes participants' head sizes, enabling more reliable comparisons of brain volume 
    across individuals.

    In the future, we hope to develop our own method for adjusting for head size rather than relying solely on 
    ATLAS scaling. Differences between natural anatomical scaling and ATLAS space could influence volumetric 
    measurements.
    """)

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


    # -------------------------------------------------------------------------
# AVERAGE BRAIN VOLUME BY CDR (mean ± SEM) — Comparing Methods
# -------------------------------------------------------------------------
st.subheader("Average Brain Volume by CDR (mean ± SEM) — Comparing Methods")

if "CDR" not in df_local.columns:
    st.warning("Dataset does not contain `CDR` column.")
else:
    try:
        fig, axes = plt.subplots(1, len(available_methods), figsize=(9 * len(available_methods), 6))
        if len(available_methods) == 1:
            axes = [axes]

        for idx, method in enumerate(available_methods):
            # Compute mean & SEM grouped by CDR
            grp = df_local.groupby("CDR")[method].agg(["mean", "sem"]).reset_index()

            axes[idx].bar(
                grp["CDR"].astype(str),
                grp["mean"],
                yerr=grp["sem"],
                capsize=6,
                color=method_colors[method]
            )

            axes[idx].set_title(method_labels[method], fontsize=16)
            axes[idx].set_xlabel("CDR", fontsize=14)
            axes[idx].set_ylabel("Average Brain Volume", fontsize=14)
            axes[idx].tick_params(axis='both', labelsize=12)

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Unable to render mean ± SEM plots: {e}")

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
    st.header('Conclusions')

    st.write("""
    Using the brain extraction algorithm, we found that participants without dementia showed 
    **greater mean brain volume** than those with dementia. However, we did not observe a 
    statistically significant difference in mean brain volume *between* different severities 
    of dementia. 

    In contrast, the deep Atropos method did not reveal significant differences in brain volume 
    between any of the participant groups. Even so, both methods demonstrated a **general trend 
    of decreasing brain volume with increasing dementia severity**, as illustrated by our 
    box-and-whisker plots. Notably, the most severe dementia group exhibited a **much narrower 
    range of brain volumes**, suggesting reduced variability in late-stage disease.
    """)

    st.subheader("Comparison of Brain Volume Estimation Methods")
    st.write("""
    When comparing the two measurement approaches, the **brain extraction model** produced more 
    interpretable results and clearer volumetric trends than the deep Atropos model. This suggests 
    that brain extraction may be better suited for studies focusing on whole-brain volume changes.
    
    In future work, we aim to compute brain volume from **raw MRI scans** rather than relying on 
    ATLAS-registered images. Because template registration involves warping, this transformation 
    may distort true anatomical differences, potentially influencing our results.
    """)

    st.subheader("Future Directions")
    st.write("""
    Future research could investigate how brain volume changes over time within individuals. 
    Analyzing longitudinal trajectories of brain volume and comparing them to changes in CDR 
    scores could offer deeper insight into disease progression.

    To support such analyses, the **OASIS-2 dataset** is an excellent candidate, as it includes 
    multiple MRI sessions taken at least one year apart for each participant. Studying temporal 
    patterns could reveal whether brain volume decline accelerates as dementia worsens or 
    whether the rate of decline varies across individuals.
    """)


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


# Radio with hidden label
page = st.sidebar.radio(
    label="",        # No visible label
    options=PAGES,
    index=PAGES.index(st.session_state.page),
)


if page != st.session_state.page:
    st.session_state.page = page

# Sidebar CSS to make pages look like tiles and remove native radio visuals
st.sidebar.markdown(
    """
    <style>
    section[data-testid="stSidebar"] svg { display: none !important; }
    section[data-testid="stSidebar"] input[type="radio"] { display: none !important; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none !important; }
    section[data-testid="stSidebar"] .stRadio {
        margin-top: -1rem !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        display: block !important;
        padding: 10px 12px !important;
        margin: 4px 0 !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.12s ease !important;
        border: 2px solid transparent !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(0,0,0,0.06) !important;
    }
    section[data-testid="stSidebar"] .stRadio label:has(input[type="radio"]:checked) {
        background: rgba(0,0,0,0.08) !important;
        font-weight: 600 !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if page == 'Overview':
    render_overview()
elif page == 'OASIS':
    render_oasis()
elif page == 'Code':
    render_code()
elif page == 'Data & Graphs':
    render_data_and_graphs()
elif page == 'Conclusions':
    render_conclusions()
elif page == 'References':
    render_references()

