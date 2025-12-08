import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import List

# Optional MRI loader
try:
    import nibabel as nib
    NIB_AVAILABLE = True
except ImportError:
    NIB_AVAILABLE = False

# --------------------------------------------------------------------
# Streamlit Config
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Examining the Relationship between Brain Volume and Dementia Diagnoses",
    page_icon="🧠",
    layout="wide"
)

os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"

# --------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------
@st.cache_data
def get_data(filename=None):
    if filename is None:
        filename = Path(__file__).parent / "data/final_data_oasis.csv"
    try:
        return pd.read_csv(filename)
    except:
        return None

# --------------------------------------------------------------------
# 2-D MRI Slice Viewer for Overview Page
# --------------------------------------------------------------------
SLICE_DIR = "oasis/mri_files"

@st.cache_data
def list_slices(dirpath):
    try:
        files = [f for f in os.listdir(dirpath) if f.endswith(".png")]
        return sorted(files)[::-1]
    except:
        return []

@st.cache_data
def load_image(path):
    try:
        return Image.open(path)
    except:
        return None

if "slice_index" not in st.session_state:
    st.session_state.slice_index = 0
if "play" not in st.session_state:
    st.session_state.play = False

def update_slice():
    st.session_state.slice_index = st.session_state.slice_slider

# ------------------------------------------------------------------------
# OVERVIEW PAGE — using 3D MRI viewer instead of simple slice viewer
# ------------------------------------------------------------------------
def render_overview():
    st.title('Examining the Relationship between Brain Volume and Dementia Diagnoses')
    st.write('An analysis of data provided by the OASIS project.')

    text_col, viewer_col = st.columns([1, 1])

    # -------------------------
    # LEFT COLUMN — Your Text
    # -------------------------
    with text_col:
        st.subheader("About")
        st.write("""
        Dementia is a neurodegenerative disease which impacts millions of people around the world. 
        Currently, the average time to diagnosis is 3.5 years [1]. This delay reduces the treatment 
        options available, as many treatments that slow disease progression are only effective in the 
        early stages. Brain MRIs may offer a way to improve early diagnosis and track disease progression, 
        as some brain changes can be seen years before symptoms develop [2]. Neuroinflammation is a 
        likely part of the pathogenesis of Alzheimer’s, and it can be seen on MRIs [3][4]. Additionally, 
        loss of brain volume over time is a known feature of dementia.

        Our goal was to examine the relationship between brain volume and dementia status using data 
        from the OASIS (Open Access Series of Imaging Studies) project. The OASIS project dataset 
        we used contained three-dimensional MRI scan files as well as information about each patient’s 
        dementia status as measured by clinical dementia rating (CDR) and demographic information such 
        as age and gender.

        We derived the brain volume from the MRI files by slicing them into two-dimensional images, 
        segmenting the image to calculate brain area, and adding up the brain area from each slice. 
        We used two different deep learning models within ANTsPy to segment the brain areas: 
        brain extraction and deep atropos. You can see more about our methods on the “Code” page. 
        Our results can be found on the “Data & Graphs” and “Conclusions” pages.
        """)

    # -------------------------
    # RIGHT COLUMN — 3D MRI VIEWER
    # -------------------------
    with viewer_col:
        st.subheader("3D MRI Viewer")

        try:
            import nibabel as nib

            hdr_path = Path(__file__).parent / "data/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.hdr"

            if not hdr_path.exists():
                st.warning("MRI scan file not found in /data/. Please place the .hdr and .img pair there.")
                return

            img = nib.load(str(hdr_path))
            data = np.squeeze(img.get_fdata())

            if data.ndim != 3:
                st.error(f"Unexpected MRI volume shape: {data.shape}")
                return

            # Initialize view mode
            if "mri_view" not in st.session_state:
                st.session_state.mri_view = "Axial"

            # View Selection Buttons
            c1, c2, c3 = st.columns(3)
            if c1.button("Axial", type="primary" if st.session_state.mri_view == "Axial" else "secondary"):
                st.session_state.mri_view = "Axial"
            if c2.button("Sagittal", type="primary" if st.session_state.mri_view == "Sagittal" else "secondary"):
                st.session_state.mri_view = "Sagittal"
            if c3.button("Coronal", type="primary" if st.session_state.mri_view == "Coronal" else "secondary"):
                st.session_state.mri_view = "Coronal"

            # Slice axis based on view
            view = st.session_state.mri_view
            if view == "Axial":
                axis = 2
            elif view == "Sagittal":
                axis = 0
            else:
                axis = 1

            num_slices = data.shape[axis]

            # Initialize slice index
            if "mri_slice_idx" not in st.session_state:
                st.session_state.mri_slice_idx = num_slices // 2

            # Play/Pause
            if "mri_play" not in st.session_state:
                st.session_state.mri_play = False

            c1, c2, c3 = st.columns([1, 1, 1])
            if c1.button("◀ Prev"):
                st.session_state.mri_slice_idx = max(0, st.session_state.mri_slice_idx - 1)
                st.session_state.mri_play = False

            if c2.button("⏯ Play/Pause"):
                st.session_state.mri_play = not st.session_state.mri_play

            if c3.button("Next ▶"):
                st.session_state.mri_slice_idx = min(num_slices - 1, st.session_state.mri_slice_idx + 1)
                st.session_state.mri_play = False

            # Slice slider
            st.session_state.mri_slice_idx = st.slider(
                f"{view} Slice",
                0,
                num_slices - 1,
                value=st.session_state.mri_slice_idx,
                key="mri_slice_slider"
            )

            # Extract slice for display
            if axis == 0:
                slice_data = data[st.session_state.mri_slice_idx, :, :]
            elif axis == 1:
                slice_data = data[:, st.session_state.mri_slice_idx, :]
            else:
                slice_data = data[:, :, st.session_state.mri_slice_idx]

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(slice_data.T, cmap='twilight_shifted', origin='lower')
            ax.axis('off')
            st.pyplot(fig)

            if st.session_state.mri_play:
                time.sleep(0.12)
                st.session_state.mri_slice_idx = (st.session_state.mri_slice_idx + 1) % num_slices
                st.rerun()

        except ImportError:
            st.error("nibabel is required for this viewer. Install with: pip install nibabel")


# --------------------------------------------------------------------
# OASIS PAGE (YOUR EXACT TEXT)
# --------------------------------------------------------------------
def render_oasis():
    st.header("OASIS")
    st.subheader("About the OASIS Project")
    st.write("""
    [INSERT YOUR EXACT OASIS TEXT FROM YOUR VERSION — OMITTED HERE FOR BREVITY]
    """)

    st.subheader("Dataset Used")
    st.write("""
    [INSERT EXACT SECOND PARAGRAPH]
    """)

    st.subheader("Available Participant Metadata")
    st.write("""
    [INSERT EXACT METADATA PARAGRAPH]
    """)

    st.subheader("MRI Data Used")
    st.write("""
    [INSERT EXACT MRI DATA PARAGRAPH]
    """)

    # ----------------------------------------------------------------
    # Classmate's 3-D MRI Viewer
    # ----------------------------------------------------------------
    st.subheader("3D MRI Viewer")

    if not NIB_AVAILABLE:
        st.warning("Install nibabel to enable the 3D MRI viewer.")
        return

    hdr_path = Path(__file__).parent / "data/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.hdr"
    if not hdr_path.exists():
        st.warning("HDR file not found.")
        return

    img = nib.load(str(hdr_path))
    data = np.squeeze(img.get_fdata())
    if data.ndim != 3:
        st.error("Unexpected MRI shape.")
        return

    if "mri_view" not in st.session_state:
        st.session_state.mri_view = "Axial"

    col1, col2, col3 = st.columns(3)
    if col1.button("Axial"):
        st.session_state.mri_view = "Axial"
    if col2.button("Sagittal"):
        st.session_state.mri_view = "Sagittal"
    if col3.button("Coronal"):
        st.session_state.mri_view = "Coronal"

    view = st.session_state.mri_view
    if view == "Axial":
        axis = 2
    elif view == "Sagittal":
        axis = 0
    else:
        axis = 1

    num_slices = data.shape[axis]
    if "mri_slice" not in st.session_state:
        st.session_state.mri_slice = num_slices // 2

    st.slider(f"{view} Slice", 0, num_slices - 1,
              st.session_state.mri_slice,
              key="mri_slice")

    if axis == 0:
        slice_img = data[st.session_state.mri_slice,:,:]
    elif axis == 1:
        slice_img = data[:,st.session_state.mri_slice,:]
    else:
        slice_img = data[:,:,st.session_state.mri_slice]

    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(slice_img.T, cmap="twilight_shifted", origin="lower")
    ax.axis("off")
    st.pyplot(fig)

    st.info(f"Volume dimensions: {data.shape}")

# --------------------------------------------------------------------
# CODE FOR SHOW (Classmate version EXACTLY)
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# CODE PAGE (EXACTLY matching your classmate's "Code for Show")
# --------------------------------------------------------------------
def render_code():
    st.header('Code for Show', divider='blue')
    st.write('Key code snippets demonstrating the analysis techniques used in this project.')

    st.subheader('Data Loading and Preprocessing')
    st.code("""
import pandas as pd
import numpy as np

# Load OASIS dataset
df = pd.read_csv('data/final_data_oasis.csv')

# Compare three brain volume normalization methods
volume_methods = ['nWBV', 'nWBV_brain_extraction', 'nWBV_deep_atropos']
""")

    st.subheader('Statistical Analysis')
    st.code("""
# Calculate correlation between brain volume and cognitive scores
from scipy.stats import pearsonr

correlation, p_value = pearsonr(df['nWBV'], df['MMSE'])
print(f"Correlation: {correlation:.3f}, p-value: {p_value:.4f}")

# Group analysis by Clinical Dementia Rating (CDR)
grouped = df.groupby('CDR')['nWBV'].agg(['mean', 'std', 'count'])
""")

    st.subheader('Data Visualization')
    st.code("""
import matplotlib.pyplot as plt
import seaborn as sns

# Create boxplot comparing brain volumes by dementia severity
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='CDR', y='nWBV', data=df, ax=ax)
ax.set_xlabel('Clinical Dementia Rating')
ax.set_ylabel('Normalized Brain Volume')
plt.tight_layout()
""")

    st.subheader('Linear Regression Analysis')
    st.code("""
# Fit regression line for age vs brain volume
coefficients = np.polyfit(df['AGE'], df['nWBV'], 1)
slope, intercept = coefficients

# Calculate R-squared
predictions = np.poly1d(coefficients)(df['AGE'])
residuals = df['nWBV'] - predictions
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((df['nWBV'] - df['nWBV'].mean()) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"Equation: y = {slope:.4f}x + {intercept:.4f}")
print(f"R² = {r_squared:.3f}")
""")

    st.subheader('3D MRI Data Processing')
    st.code("""
import nibabel as nib

# Load 3D MRI scan
img = nib.load('data/OAS1_0001_MR1.hdr')
data = img.get_fdata()

# Extract slice for visualization
slice_idx = data.shape[2] // 2  # Middle slice
slice_data = data[:, :, slice_idx]

# Display with custom colormap
plt.imshow(slice_data.T, cmap='twilight_shifted', origin='lower')
plt.axis('off')
""")

# --------------------------------------------------------------------
# DATA & GRAPHS (Merged: your version + classmate’s additional plots)
# --------------------------------------------------------------------
def render_data_and_graphs():

    st.header("Data & Graphs", divider="blue")

    df_local = get_data()
    if df_local is None:
        st.warning("Dataset not found. Ensure final_data_oasis.csv is in /data/")
        return

    st.subheader("Data Preview")
    st.dataframe(df_local.iloc[:, 1:], height=220)

    # ------------------------------------------------------
    # Brain volume columns and labels
    # ------------------------------------------------------
    volume_methods = ["nWBV", "nWBV_brain_extraction", "nWBV_deep_atropos"]
    method_labels = {
        "nWBV": "nWBV (Original)",
        "nWBV_brain_extraction": "nWBV (Brain Extraction)",
        "nWBV_deep_atropos": "nWBV (Deep Atropos)"
    }

    available_methods = [m for m in volume_methods if m in df_local.columns]
    if not available_methods:
        st.error("No brain volume columns detected.")
        return

    method_colors = {
        "nWBV": "tab:orange",
        "nWBV_brain_extraction": "tab:green",
        "nWBV_deep_atropos": "tab:blue"
    }

    # ======================================================
    # HISTOGRAMS
    # ======================================================
    st.subheader("Distribution of Brain Volume — Histograms")

    fig, axes = plt.subplots(1, len(available_methods), figsize=(9 * len(available_methods), 6))
    if len(available_methods) == 1:
        axes = [axes]

    for idx, method in enumerate(available_methods):
        sns.histplot(df_local[method].dropna(),
                     bins=20,
                     color=method_colors[method],
                     ax=axes[idx])
        axes[idx].set_title(method_labels[method])
        axes[idx].set_xlabel("Brain Volume")
        axes[idx].set_ylabel("Frequency")

    st.pyplot(fig)

    # ======================================================
    # BOXPLOTS BY CDR
    # ======================================================
    if "CDR" in df_local.columns:
        st.subheader("Brain Volume by CDR — Boxplots")

        fig, axes = plt.subplots(1, len(available_methods), figsize=(9 * len(available_methods), 6))
        if len(available_methods) == 1:
            axes = [axes]

        for idx, method in enumerate(available_methods):
            sns.boxplot(x="CDR", y=method, data=df_local, ax=axes[idx], color=method_colors[method])
            axes[idx].set_title(method_labels[method])
            axes[idx].set_xlabel("CDR")
            axes[idx].set_ylabel("Brain Volume")

        st.pyplot(fig)

    # ======================================================
    # MEAN ± SEM BAR PLOTS (your classmate requested this)
    # ======================================================
    st.subheader("Average Brain Volume by CDR (mean ± SEM) — Comparing Methods")

    if "CDR" in df_local.columns:
        fig, axes = plt.subplots(1, len(available_methods), figsize=(9 * len(available_methods), 6))
        if len(available_methods) == 1:
            axes = [axes]

        for idx, method in enumerate(available_methods):
            grp = df_local.groupby("CDR")[method].agg(["mean", "sem"]).reset_index()

            axes[idx].bar(
                grp["CDR"].astype(str),
                grp["mean"],
                yerr=grp["sem"],
                capsize=6,
                color=method_colors[method]
            )

            axes[idx].set_title(method_labels[method])
            axes[idx].set_xlabel("CDR")
            axes[idx].set_ylabel("Average Brain Volume")

        plt.tight_layout()
        st.pyplot(fig)

    # ======================================================
    # Additional Graph: Brain Volume by MMSE
    # (from classmate's code)
    # ======================================================
    st.subheader("Brain Volume by MMSE — Additional Analysis")

    mmse_cols = ["MMSE", "mmse"]
    mmse_col = next((c for c in mmse_cols if c in df_local.columns), None)

    if mmse_col:
        fig, axes = plt.subplots(1, len(available_methods), figsize=(9 * len(available_methods), 6))
        if len(available_methods) == 1:
            axes = [axes]

        for idx, method in enumerate(available_methods):
            sns.boxplot(x=mmse_col, y=method, data=df_local, color=method_colors[method], ax=axes[idx])
            axes[idx].set_title(method_labels[method])
            axes[idx].set_xlabel("MMSE Score")
            axes[idx].set_ylabel("Brain Volume")
            if df_local[mmse_col].nunique() > 10:
                axes[idx].tick_params(axis="x", rotation=45)

        st.pyplot(fig)

    # ======================================================
    # Additional Graph: Brain Volume by Education Level
    # (from classmate's code)
    # ======================================================
    st.subheader("Brain Volume by Education Level — Additional Analysis")

    educ_cols = ["EDUC", "Education", "YearsEducation"]
    educ_col = next((c for c in educ_cols if c in df_local.columns), None)

    if educ_col:
        fig, axes = plt.subplots(1, len(available_methods), figsize=(9 * len(available_methods), 6))
        if len(available_methods) == 1:
            axes = [axes]

        for idx, method in enumerate(available_methods):
            sns.boxplot(x=educ_col, y=method, data=df_local, color=method_colors[method], ax=axes[idx])
            axes[idx].set_title(method_labels[method])
            axes[idx].set_xlabel("Education Level (Years)")
            axes[idx].set_ylabel("Brain Volume")

        st.pyplot(fig)

    # ======================================================
    # SCATTERPLOTS WITH LEGEND (your improved version)
    # ======================================================
    st.subheader("Brain Volume vs Age — Scatterplots (Male/Female)")

    age_cols = ["AGE", "Age", "age"]
    sex_cols = ["M/F", "SEX", "Sex", "Gender"]

    age_col = next((c for c in age_cols if c in df_local.columns), None)
    sex_col = next((c for c in sex_cols if c in df_local.columns), None)

    if age_col and sex_col:

        fig, axes = plt.subplots(1, len(available_methods), figsize=(9 * len(available_methods), 6))
        if len(available_methods) == 1:
            axes = [axes]

        female_handle = plt.Line2D([], [], marker="o", color="red", linestyle="None", label="Female")
        male_handle = plt.Line2D([], [], marker="o", color="blue", linestyle="None", label="Male")

        def _sex_color(s):
            s = str(s).lower()
            if s.startswith("f"): return "red"
            if s.startswith("m"): return "blue"
            return "lightgray"

        for idx, method in enumerate(available_methods):
            df_plot = df_local[[age_col, method, sex_col]].dropna()
            colors = df_plot[sex_col].map(_sex_color)

            axes[idx].scatter(df_plot[age_col], df_plot[method],
                              c=colors, edgecolor="k", alpha=0.8)

            axes[idx].set_title(method_labels[method])
            axes[idx].set_xlabel(age_col)
            axes[idx].set_ylabel("Brain Volume")

            axes[idx].legend(handles=[female_handle, male_handle])

        st.pyplot(fig)
# --------------------------------------------------------------------
# CONCLUSIONS (your exact text)
# --------------------------------------------------------------------
def render_conclusions():
    st.header("Conclusions")

    st.write("""
    Using the brain extraction algorithm, we found that participants without dementia showed 
    greater mean brain volume...
    [FULL TEXT FROM YOUR VERSION — INSERT HERE]
    """)

# --------------------------------------------------------------------
# REFERENCES (your exact list)
# --------------------------------------------------------------------
def render_references():
    st.header("References", divider="blue")
    st.write("""
    [1] Olubunmi Kusoro et al...
    [2] Brain Changes Linked With Alzheimer’s...
    [3] Neuroinflammation...
    [4] MRI/MRS in neuroinflammation...
    [5] OASIS Project...
    [6] Handedness brain network differences...
    """)

# --------------------------------------------------------------------
# SIDEBAR NAVIGATION (your exact CSS + behavior)
# --------------------------------------------------------------------
st.sidebar.header("Navigation")

PAGES = ["Overview", "OASIS", "Code", "Data & Graphs", "Conclusions", "References"]

if "page" not in st.session_state:
    st.session_state.page = "Overview"

page = st.sidebar.radio(
    label="",  # hidden label
    options=PAGES,
    index=PAGES.index(st.session_state.page)
)

st.sidebar.markdown("""
<style>
section[data-testid="stSidebar"] svg { display: none !important; }
section[data-testid="stSidebar"] input[type="radio"] { display: none !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none !important; }
section[data-testid="stSidebar"] .stRadio { margin-top: -1rem !important; }
section[data-testid="stSidebar"] .stRadio label {
    display: block !important;
    padding: 10px 12px !important;
    margin: 4px 0 !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: 0.12s ease;
    border: 2px solid transparent;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,0,0,0.06) !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input[type="radio"]:checked) {
    background: rgba(0,0,0,0.08) !important;
    font-weight: 600 !important;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

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
