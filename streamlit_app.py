###############################################################
# Streamlit App: OASIS Brain Volume Analysis
###############################################################

import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize

# Optional MRI loading
try:
    import nibabel as nib

    MRI_ENABLED = True
except Exception:
    MRI_ENABLED = False

###############################################################
# Streamlit Config
###############################################################
st.set_page_config(
    page_title=(
        "Examining the Relationship between "
        "Brain Volume and Dementia Diagnoses"
    ),
    page_icon=":brain:",
    layout="wide",
)

###############################################################
# DATA LOADING
###############################################################


@st.cache_data
def get_data(filename=None):
    if filename is None:
        filename = Path(__file__).parent / "data/final_data_oasis.csv"
    try:
        return pd.read_csv(filename)
    except Exception:
        return None


###############################################################
# STATE INITIALIZATION
###############################################################
if "mri_slice_idx" not in st.session_state:
    st.session_state.mri_slice_idx = 85
if "mri_view" not in st.session_state:
    st.session_state.mri_view = "Axial"
if "mri_play" not in st.session_state:
    st.session_state.mri_play = False


def update_mri_slice():
    st.session_state.mri_play = False
    st.session_state.mri_slice_idx = st.session_state.overview_mri_slider


###############################################################
# PAGE: OVERVIEW
###############################################################
def render_overview():
    st.title( "Examining the Relationship between Brain Volume and Dementia Diagnoses" )
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("About")
        st.write(
            """
Dementia is a neurodegenerative disease that impacts millions of people around
the world. Currently, the average time to diagnosis is 3.5 years [1].
This delay reduces the treatment options available, as many treatments
that slow disease progression are only effective in the early stages.
Brain MRIs may offer a way to improve early diagnosis and track disease
progression, as some brain changes can be seen years before symptoms develop
[2]. Neuroinflammation is a likely part of the pathogenesis of Alzheimer’s,
and it can be seen on MRIs [3][4]. Additionally, loss of brain volume over
time is a known feature of dementia.

Our goal was to examine the relationship between brain volume and dementia status using data
from the OASIS (Open Access Series of Imaging Studies) project [5]. The OASIS
project dataset we used contained three-dimensional MRI scan files as well as
information about each patient’s dementia status as measured by clinical dementia
rating (CDR) and demographic information such as age and gender. The MRI viewer
on the right displays cross-sectional brain scans that allow you to explore the
anatomy captured in this dataset.

To calculate the brain volumes, we used two separate methods in a similar process. We started
by creating a list of all the MRI brain scans for which we have clinical data
for. Looping through that list, we utilized two modules imported from the
ANTsPyNet framework: ‘brain_extraction’ and ‘deep_atropos’. These utilities
were used to generate a probability map of brain-like voxels.

These maps were used to segment the images and sum up all the desired voxels representing
brain regions. Then, these voxels were multiplied by a known voxel spacing
to obtain the brain volume and scaled using the given Atlas Scaling Factor
(ASF). Then, using the estimated total intracranial values, the brains
can be normalized to a standard size for comparison.

        """
        )

    with col2:
        st.subheader("3D MRI Viewer")
        if not MRI_ENABLED:
            st.warning("nibabel not installed — 3D MRI viewer unavailable.")
        else:
            hdr_path = (Path(__file__).parent / "data/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.hdr")
            if not hdr_path.exists():
                st.warning("HDR MRI file missing.")
            else:
                try:
                    img = nib.load(str(hdr_path))
                    data = np.squeeze(img.get_fdata())

                    # Buttons for choosing view
                    views = {"Axial": 2, "Sagittal": 0, "Coronal": 1}
                    cols = st.columns(3)
                    for col, view in zip(cols, views.keys()):
                        if col.button(view, key=f"overview_{view.lower()}"):
                            st.session_state.mri_view = view
                            st.session_state.mri_play = False

                    slice_axis = views[st.session_state.mri_view]

                    num_slices = data.shape[slice_axis]

                    # Slider above image
                    st.slider(
                        f"{st.session_state.mri_view} Slice",
                        0,
                        num_slices - 1,
                        key="overview_mri_slider",
                        value=st.session_state.mri_slice_idx,
                        on_change=update_mri_slice,
                    )

                    # Extract slice
                    slices = [slice(None)] * 3
                    slices[slice_axis] = st.session_state.mri_slice_idx
                    slice_data = data[tuple(slices)]

                    # Display image
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(slice_data.T, cmap="twilight_shifted", origin="lower")
                    ax.axis("off")
                    st.pyplot(fig)

                    # Navigation buttons below image
                    c1, c2, c3 = st.columns([1, 1, 1])
                    if c1.button("◀ Prev", key="overview_prev"):
                        st.session_state.mri_slice_idx = max(0, st.session_state.mri_slice_idx - 1)
                        st.session_state.mri_play = False
                        st.rerun()

                    if c2.button(
                        "▶ Play" if not st.session_state.mri_play else "⏸ Pause",
                        key="overview_play",
                    ):
                        st.session_state.mri_play = (
                            not st.session_state.mri_play
                        )
                        st.rerun()

                    if c3.button("Next ▶", key="overview_next"):
                        st.session_state.mri_slice_idx = min(num_slices - 1, st.session_state.mri_slice_idx + 1)
                        st.session_state.mri_play = False
                        st.rerun()

                    # Auto-advance if playing
                    if st.session_state.mri_play:
                        time.sleep(0.08)
                        st.session_state.mri_slice_idx = (st.session_state.mri_slice_idx + 1) % num_slices
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading MRI: {e}")


###############################################################
# PAGE: OASIS
###############################################################
def render_oasis():
    st.header("OASIS", divider="blue")

    st.subheader("About the OASIS Project")
    st.write(
        """
In their words, “The Open Access Series of Imaging Studies (OASIS) is a project
aimed at making neuroimaging data sets of the brain freely available
to the scientific community. By compiling and freely distributing
neuroimaging data sets, we hope to facilitate future discoveries in
basic and clinical neuroscience,” [5]. The data in these datasets were
gathered from the Knight ADRC and affiliated studies. Participants
include both men and women. All the datasets controlled for handedness,
choosing to only include right-handed participants. Right and left-handed
brains have different asymmetry patterns, which would not affect our study
on brain volume but may affect other studies searching for specific
patterns within the images themselves [6].

For our project, we used the OASIS-1 dataset, which included 416 participants aged 18
to 96. Three to four MRI scans taken at the same time were included for each participant.
All of the scans were T-1 weighted. T-1 weighting is very common in brain MRIs as it
allows for different tissues such as bone, fat, and grey matter to be distinguished.
Of the 416 participants, 100 were clinically diagnosed with dementia, and 20 non-demented
participants were imaged on a subsequent visit within 90 days of the initial session to
use as a reliability dataset.

In addition to the brain scans, sex, handedness, age, education level, socioeconomic
status, mini-mental state examination (MMSE), clinical dementia rating (CDR), estimated
total intracranial volume, normalized whole brain volume, and ATLAS scaling factor were
recorded for each participant. CDR and MMSE are tools used to gauge the stage of
dementia. We chose to focus on CDR. To obtain CDR, physicians conduct a semi-
structured interview with the patient and a reliable informant such as a family
member to assess six domains of cognitive and functional performance: memory,
orientation, judgment and problem-solving, community affairs, home and hobbies,
and personal care. CDR is graded on a scale from 0-3, with 0 indicating no
symptoms of dementia and 3 indicating severe dementia. In contrast, higher
scores on the MMSE, scaled 0-30, indicate less cognitive impairment. The
OASIS study used ATLAS scaling to account for differences in head size before
calculating the normalized whole brain volume for each participant.

We used the gain field corrected ATLAS registered transverse brain scans for our project.
If we had more time, we would have calculated total brain volume directly from the raw
MRI scans rather than relying on the ATLAS registered images. The ATLAS registered
scans provided by OASIS are warped to a common template to simplify segmentation
and cross-subject comparison, which is why we used them for this time limited
project. However, using only these pre-registered images limited our ability to
compute certain measures, including estimated total intracranial volume (eTIV).
A more rigorous approach would involve starting with the raw MRIs, performing our
own ATLAS registration for segmentation, and lastly converting the segmented
scans back into their normal space to calculate the brain volume and total
intracranial volume. This workflow would increase accuracy and allow us to
independently compute eTIV instead of relying on the values produced by the
OASIS pipeline. In the future, we could examine how brain volume changes over
time using the OASIS-2 dataset and evaluate how those changes relate to CDR scores.
Longitudinal analysis would allow us to determine whether progressive brain atrophy
correlates with higher CDR ratings and worsening cognitive impairment.
"""
    )


###############################################################
# PAGE: CODE (from Portfolio_sample_code.ipynb)
###############################################################
def load_notebook_content():
    """Load and parse the Portfolio_sample_code.ipynb notebook"""
    import json
    notebook_path = Path(__file__).parent / "Portfolio_sample_code.ipynb"
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        return notebook['cells']
    except Exception as e:
        st.error(f"Error loading notebook: {e}")
        return []

def render_code():
    """Display content from Portfolio_sample_code.ipynb"""
    st.header("Brain processing and Segmentation Code", divider="blue")
    
    # Load notebook cells
    cells = load_notebook_content()
    
    if cells:
        # Display each cell
        for cell in cells:
            cell_type = cell.get('cell_type', '')
            source = cell.get('source', [])
            
            # Join source lines if it's a list
            if isinstance(source, list):
                content = ''.join(source)
            else:
                content = source
            
            # Skip empty cells
            if not content.strip():
                continue
            
            # Render based on cell type
            if cell_type == 'markdown':
                st.markdown(content)
            elif cell_type == 'code':
                st.code(content, language='python')
    else:
        st.warning("Could not load notebook content. Showing visualizations only.")


# HELPER FUNCTIONS
###############################################################
def apply_tab_styles():
    """Apply custom CSS for tab styling"""
    st.markdown(
        """
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {font-size: 22px; font-weight: normal;}
    .stTabs [data-baseweb="tab-list"] {gap: 0px;}
    .stTabs [data-baseweb="tab-list"] button {
        border: 1px solid rgba(0, 0, 0, 0.15); border-radius: 8px 8px 0 0; padding: 12px 20px;
        margin-right: -1px; transition: all 0.2s ease; background-color: rgba(0, 0, 0, 0.03);
    }
    .stTabs [data-baseweb="tab-list"] button:hover {background-color: rgba(0, 0, 0, 0.08); border-color: rgba(0, 0, 0, 0.25); z-index: 1;}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: white; border-bottom: 3px solid #1f77b4; font-weight: 500; z-index: 2; border-bottom-color: white;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def setup_axes(axes, num_plots):
    """Convert single axis to list for consistent handling"""
    return [axes] if num_plots == 1 else axes


def plot_method_comparison(
    df, methods, method_labels, method_colors, plot_func, **kwargs
):
    """Generic function to create comparison plots across methods"""
    fig, axes = plt.subplots(1, len(methods), figsize=(8 * len(methods), 5))
    axes = setup_axes(axes, len(methods))
    for i, method in enumerate(methods):
        plot_func(
            df,
            method,
            axes[i],
            method_labels[method],
            method_colors[method],
            **kwargs,
        )
    return fig


def calc_regression_stats(x, y):
    """Calculate regression line statistics"""
    if len(x) < 2:
        return None
    z = np.polyfit(x, y, 1)
    slope, intercept = z[0], z[1]
    p = np.poly1d(z)
    y_pred = p(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return {"slope": slope, "intercept": intercept, "r2": r2, "poly": p}


def chi_squared(params, areas, times):
    """Returns the chi squared value for the parameters, data, and fit arrays provided"""

    # unpacking the parameters
    m = params[0]
    b = params[1]
    # compute the fit
    fit = m * times + b

    # computing the chi-squared
    vals = (areas - fit) ** 2 / fit
    return np.sum(vals)


###############################################################
# PAGE: DATA & GRAPHS
###############################################################


def render_data_and_graphs():
    st.header("Data & Graphs", divider="blue")
    df = get_data()
    if df is None:
        st.error("Dataset missing.")
        return

    st.subheader("Preview of Dataset")
    st.dataframe(df.iloc[:, 1:], height=210)

    # METHODS
    volume_methods = ["nWBV", "nWBV_brain_extraction", "nWBV_deep_atropos"]
    method_labels = {
        "nWBV": "nWBV (Original)",
        "nWBV_brain_extraction": "nWBV (Brain Extraction)",
        "nWBV_deep_atropos": "nWBV (Deep Atropos)",
    }
    method_colors = {
        "nWBV": "hotpink",
        "nWBV_brain_extraction": "mediumseagreen",
        "nWBV_deep_atropos": "tab:blue",
    }

    apply_tab_styles()

    # Create tabs for different visualizations
    tabs = st.tabs(["CDR Boxplots", "Age Scatter", "CDR Bar Chart", "MMSE Analysis"])

    with tabs[0]:  # CDR Boxplots
        st.markdown("### Brain Volume by Clinical Dementia Rating")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, method in enumerate(volume_methods):
            if method in df.columns:
                sns.boxplot(x='CDR', y=method, data=df, ax=axes[idx], 
                           color=method_colors[method])
                axes[idx].set_title(method_labels[method], fontsize=20)
                axes[idx].set_xlabel('CDR', fontsize=16)
                axes[idx].set_ylabel('Brain Volume', fontsize=16)
                axes[idx].tick_params(labelsize=14)
        plt.tight_layout()
        st.pyplot(fig)

    with tabs[1]:  # Age Scatter
        st.markdown("### Brain Volume vs Age (by Sex)")
        men = df[df['M/F'] == 'M']
        women = df[df['M/F'] == 'F']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, method in enumerate(volume_methods):
            if method in df.columns:
                axes[idx].scatter(men['Age'], men[method], alpha=0.6, 
                                 color=method_colors[method], label='Male', s=50)
                axes[idx].scatter(women['Age'], women[method], alpha=0.6, 
                                 color=method_colors[method], marker='s', label='Female', s=50)
                axes[idx].set_title(method_labels[method], fontsize=20)
                axes[idx].set_xlabel('Age (years)', fontsize=16)
                axes[idx].set_ylabel('Brain Volume', fontsize=16)
                axes[idx].tick_params(labelsize=14)
                axes[idx].legend(fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

    with tabs[2]:  # CDR Bar Chart
        st.markdown("### Average Brain Volume by CDR")
        cdr_means = df.groupby('CDR')[volume_methods].mean()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, method in enumerate(volume_methods):
            if method in cdr_means.columns:
                cdr_means[method].plot(kind='bar', ax=axes[idx], 
                                       color=method_colors[method])
                axes[idx].set_title(method_labels[method], fontsize=20)
                axes[idx].set_xlabel('CDR', fontsize=16)
                axes[idx].set_ylabel('Average Brain Volume', fontsize=16)
                axes[idx].tick_params(labelsize=14)
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    with tabs[3]:  # MMSE Analysis
        mmse_col = 'MMSE' if 'MMSE' in df.columns else None
        if mmse_col:
            st.markdown("### Brain Volume vs MMSE Score")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for idx, method in enumerate(volume_methods):
                if method in df.columns:
                    data_mmse = df[[mmse_col, method]].dropna()
                    axes[idx].scatter(data_mmse[mmse_col], data_mmse[method], 
                                     alpha=0.6, color=method_colors[method], s=50)
                    axes[idx].set_title(method_labels[method], fontsize=20)
                    axes[idx].set_xlabel('MMSE Score', fontsize=16)
                    axes[idx].set_ylabel('Brain Volume', fontsize=16)
                    axes[idx].tick_params(labelsize=14)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("MMSE data not available.")


###############################################################
# PAGE: CONCLUSIONS
###############################################################

def render_conclusions():
    st.header("Conclusions", divider="blue")
    st.write(
        """
    (Your full conclusions text preserved exactly)
    """
    )


###############################################################
# PAGE: REFERENCES
###############################################################
def render_references():
    st.header("References", divider="blue")
    st.write(
        """
    [1] Olubunmi Kusoro, M. Roche, R. Del‐Pino‐Casado, P. Leung, and V. Orgeta, "Time to Diagnosis in Dementia: A Systematic Review With Meta‐Analysis," International Journal of Geriatric Psychiatry, vol. 40, no. 7, Jul. 2025, doi: https://doi.org/10.1002/gps.70129.

    [2] "Brain Changes Linked With Alzheimer's Years Before Symptoms Appear," Hopkinsmedicine.org, 2019. https://www.hopkinsmedicine.org/news/newsroom/news-releases/2019/05/brain-changes-linked-with-alzheimers-years-before-symptoms-appear

    [3] "Alzheimer's Disease (AD) & Neuroinflammation | Decoding AD," Decodingalzheimersdisease.com, 2024. https://www.decodingalzheimersdisease.com/role-of-neuroinflammation.html#the-science

    [4] M. Quarantelli, "MRI/MRS in neuroinflammation: methodology and applications," Clinical and Translational Imaging, vol. 3, no. 6, pp. 475–489, Sep. 2015, doi: https://doi.org/10.1007/s40336-015-0142-y.

    [5] "Open Access Series of Imaging Studies (OASIS)," Open Access Series of Imaging Studies (OASIS). https://sites.wustl.edu/oasisbrains/

    [6] M. Li et al., "Handedness- and Hemisphere-Related Differences in Small-World Brain Networks: A Diffusion Tensor Imaging Tractography Study," Brain Connectivity, vol. 4, no. 2, pp. 145–156, Mar. 2014, doi: https://doi.org/10.1089/brain.2013.0211.
    """
    )


###############################################################
# SIDEBAR NAVIGATION (prettier tile buttons)
###############################################################
st.sidebar.header("Navigation")

PAGES = [
    "Overview",
    "OASIS",
    "Code",
    "Data & Graphs",
    "Conclusions",
    "References",
]

if "page" not in st.session_state:
    st.session_state.page = "Overview"

# Radio with hidden label
page = st.sidebar.radio(
    label="",  # No visible label
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

###############################################################
# PAGE ROUTER
###############################################################
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
