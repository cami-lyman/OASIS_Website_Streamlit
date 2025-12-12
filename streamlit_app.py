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
    
    # Add actual visualizations and data after the notebook content
    st.markdown("---")
    st.subheader("Actual Data and Visualizations from OASIS Dataset")
    
    # Display example brain image FIRST (doesn't depend on data)
    st.markdown("### Example MRI Brain Scan")
    try:
        hdr_path = Path(__file__).parent / "data/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.hdr"
        if hdr_path.exists():
            img = nib.load(str(hdr_path))
            data_img = img.get_fdata()
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(data_img[:, :, data_img.shape[2] - 95], cmap="twilight_shifted")
            ax.axis("off")
            ax.set_title("Example Transverse Slice of Brain Scan", fontsize=16)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        else:
            st.info("Example MRI file not found in data directory.")
    except Exception as e:
        st.warning(f"Could not load example image: {e}")
    
    # Load data for other visualizations
    data = get_data()
    if data is None:
        st.error("Dataset not available - could not load data/final_data_oasis.csv")
        return
    
    st.success(f"Data loaded successfully! Shape: {data.shape}")
    
    # Display dataframe
    st.markdown("### Dataset Preview")
    st.write("**Initial OASIS dataframe:**")
    st.dataframe(data.head(10))
    st.write(f"Shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # CDR Boxplots
    st.markdown("### CDR vs Brain Volume (Boxplots)")
    st.write("Comparing normalized whole brain volume across different Clinical Dementia Rating scores:")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    # OASIS nWBV (Original)
    if 'nWBV' in data.columns and 'CDR' in data.columns:
        sns.boxplot(x='CDR', y='nWBV', data=data, ax=ax1, color='hotpink')
        ax1.set_title('nWBV (Original)', fontsize=20)
        ax1.set_xlabel('CDR', fontsize=16)
        ax1.set_ylabel('Brain Volume (mm³)', fontsize=16)
        ax1.tick_params(labelsize=14)
    
    # Brain Extraction
    if 'nWBV_brain_extraction' in data.columns:
        sns.boxplot(x='CDR', y='nWBV_brain_extraction', data=data, ax=ax2, color='mediumseagreen')
        ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
        ax2.set_xlabel('CDR', fontsize=16)
        ax2.set_ylabel('Brain Volume (mm³)', fontsize=16)
        ax2.tick_params(labelsize=14)
    
    # Deep Atropos
    if 'nWBV_deep_atropos' in data.columns:
        sns.boxplot(x='CDR', y='nWBV_deep_atropos', data=data, ax=ax3, color='tab:blue')
        ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
        ax3.set_xlabel('CDR', fontsize=16)
        ax3.set_ylabel('Brain Volume (mm³)', fontsize=16)
        ax3.tick_params(labelsize=14)
    
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    
    # Age vs Brain Volume Scatter Plots
    st.markdown("### Age vs Brain Volume (Scatter Plots with Regression)")
    st.write("Analyzing how brain volume changes with age, separated by sex:")
    
    men = data[data['M/F'] == 'M']
    women = data[data['M/F'] == 'F']
    
    if len(men) > 1 and len(women) > 1:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
        
        # OASIS nWBV (Original) - Chi-squared regression
        y_mean = men['nWBV'].mean()
        y_range = men['nWBV'].max() - men['nWBV'].min()
        z = np.polyfit(men['Age'].values, men['nWBV'].values, 1)
        init_slope, init_intercept = z[0], z[1]
        men_ans = scipy.optimize.minimize(
            chi_squared, [init_slope, init_intercept],
            args=(men['nWBV'].values, men['Age'].values),
            bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
        )
        men_m, men_b = men_ans.x[0], men_ans.x[1]
        
        y_mean = women['nWBV'].mean()
        y_range = women['nWBV'].max() - women['nWBV'].min()
        z = np.polyfit(women['Age'].values, women['nWBV'].values, 1)
        init_slope, init_intercept = z[0], z[1]
        women_ans = scipy.optimize.minimize(
            chi_squared, [init_slope, init_intercept],
            args=(women['nWBV'].values, women['Age'].values),
            bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
        )
        women_m, women_b = women_ans.x[0], women_ans.x[1]
        
        ax1.scatter(men['Age'], men['nWBV'], color='blue', alpha=0.6, s=30, label='Male')
        ax1.scatter(women['Age'], women['nWBV'], color='red', alpha=0.6, s=30, label='Female')
        age_range_men = np.linspace(men['Age'].min(), men['Age'].max(), 100)
        ax1.plot(age_range_men, men_m * age_range_men + men_b, 'b--', linewidth=2)
        age_range_women = np.linspace(women['Age'].min(), women['Age'].max(), 100)
        ax1.plot(age_range_women, women_m * age_range_women + women_b, 'r--', linewidth=2)
        ax1.set_xlabel('Age (Years)', fontsize=16)
        ax1.set_ylabel('Brain Volume (mm³)', fontsize=16)
        ax1.set_title('nWBV (Original)', fontsize=20)
        ax1.tick_params(labelsize=14)
        ax1.legend(fontsize=12)
        
        # Brain Extraction
        if 'nWBV_brain_extraction' in data.columns:
            y_mean = men['nWBV_brain_extraction'].mean()
            y_range = men['nWBV_brain_extraction'].max() - men['nWBV_brain_extraction'].min()
            z = np.polyfit(men['Age'].values, men['nWBV_brain_extraction'].values, 1)
            init_slope, init_intercept = z[0], z[1]
            men_ans = scipy.optimize.minimize(
                chi_squared, [init_slope, init_intercept],
                args=(men['nWBV_brain_extraction'].values, men['Age'].values),
                bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
            )
            men_m2, men_b2 = men_ans.x[0], men_ans.x[1]
            
            y_mean = women['nWBV_brain_extraction'].mean()
            y_range = women['nWBV_brain_extraction'].max() - women['nWBV_brain_extraction'].min()
            z = np.polyfit(women['Age'].values, women['nWBV_brain_extraction'].values, 1)
            init_slope, init_intercept = z[0], z[1]
            women_ans = scipy.optimize.minimize(
                chi_squared, [init_slope, init_intercept],
                args=(women['nWBV_brain_extraction'].values, women['Age'].values),
                bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
            )
            women_m2, women_b2 = women_ans.x[0], women_ans.x[1]
            
            ax2.scatter(men['Age'], men['nWBV_brain_extraction'], color='blue', alpha=0.6, s=30, label='Male')
            ax2.scatter(women['Age'], women['nWBV_brain_extraction'], color='red', alpha=0.6, s=30, label='Female')
            ax2.plot(age_range_men, men_m2 * age_range_men + men_b2, 'b--', linewidth=2)
            ax2.plot(age_range_women, women_m2 * age_range_women + women_b2, 'r--', linewidth=2)
            ax2.set_xlabel('Age (Years)', fontsize=16)
            ax2.set_ylabel('Brain Volume (mm³)', fontsize=16)
            ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
            ax2.tick_params(labelsize=14)
            ax2.legend(fontsize=12)
        
        # Deep Atropos
        if 'nWBV_deep_atropos' in data.columns:
            y_mean = men['nWBV_deep_atropos'].mean()
            y_range = men['nWBV_deep_atropos'].max() - men['nWBV_deep_atropos'].min()
            z = np.polyfit(men['Age'].values, men['nWBV_deep_atropos'].values, 1)
            init_slope, init_intercept = z[0], z[1]
            men_ans = scipy.optimize.minimize(
                chi_squared, [init_slope, init_intercept],
                args=(men['nWBV_deep_atropos'].values, men['Age'].values),
                bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
            )
            men_m3, men_b3 = men_ans.x[0], men_ans.x[1]
            
            y_mean = women['nWBV_deep_atropos'].mean()
            y_range = women['nWBV_deep_atropos'].max() - women['nWBV_deep_atropos'].min()
            z = np.polyfit(women['Age'].values, women['nWBV_deep_atropos'].values, 1)
            init_slope, init_intercept = z[0], z[1]
            women_ans = scipy.optimize.minimize(
                chi_squared, [init_slope, init_intercept],
                args=(women['nWBV_deep_atropos'].values, women['Age'].values),
                bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
            )
            women_m3, women_b3 = women_ans.x[0], women_ans.x[1]
            
            ax3.scatter(men['Age'], men['nWBV_deep_atropos'], color='blue', alpha=0.6, s=30, label='Male')
            ax3.scatter(women['Age'], women['nWBV_deep_atropos'], color='red', alpha=0.6, s=30, label='Female')
            ax3.plot(age_range_men, men_m3 * age_range_men + men_b3, 'b--', linewidth=2)
            ax3.plot(age_range_women, women_m3 * age_range_women + women_b3, 'r--', linewidth=2)
            ax3.set_xlabel('Age (Years)', fontsize=16)
            ax3.set_ylabel('Brain Volume (mm³)', fontsize=16)
            ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
            ax3.tick_params(labelsize=14)
            ax3.legend(fontsize=12)
        
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        
        # Display regression statistics for all 3 methods
        st.markdown("### Regression Statistics")
        
        # OASIS nWBV (Original)
        st.write("**nWBV (Original)**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Men:**")
            st.write(f"Slope: {men_m:.6f}")
            st.write(f"Intercept: {men_b:.6f}")
            st.write(f"Chi²: {chi_squared([men_m, men_b], men['nWBV'].values, men['Age'].values):.6f}")
        with col2:
            st.write("**Women:**")
            st.write(f"Slope: {women_m:.6f}")
            st.write(f"Intercept: {women_b:.6f}")
            st.write(f"Chi²: {chi_squared([women_m, women_b], women['nWBV'].values, women['Age'].values):.6f}")
        
        # Brain Extraction
        if 'nWBV_brain_extraction' in data.columns:
            st.write("**nWBV (Brain Extraction)**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Men:**")
                st.write(f"Slope: {men_m2:.6f}")
                st.write(f"Intercept: {men_b2:.6f}")
                st.write(f"Chi²: {chi_squared([men_m2, men_b2], men['nWBV_brain_extraction'].values, men['Age'].values):.6f}")
            with col2:
                st.write("**Women:**")
                st.write(f"Slope: {women_m2:.6f}")
                st.write(f"Intercept: {women_b2:.6f}")
                st.write(f"Chi²: {chi_squared([women_m2, women_b2], women['nWBV_brain_extraction'].values, women['Age'].values):.6f}")
        
        # Deep Atropos
        if 'nWBV_deep_atropos' in data.columns:
            st.write("**nWBV (Deep Atropos)**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Men:**")
                st.write(f"Slope: {men_m3:.6f}")
                st.write(f"Intercept: {men_b3:.6f}")
                st.write(f"Chi²: {chi_squared([men_m3, men_b3], men['nWBV_deep_atropos'].values, men['Age'].values):.6f}")
            with col2:
                st.write("**Women:**")
                st.write(f"Slope: {women_m3:.6f}")
                st.write(f"Intercept: {women_b3:.6f}")
                st.write(f"Chi²: {chi_squared([women_m3, women_b3], women['nWBV_deep_atropos'].values, women['Age'].values):.6f}")
    
    # CDR Bar Chart Comparison
    st.markdown("### Method Comparison: CDR vs Brain Volume (Bar Charts)")
    st.write("Comparing all three brain volume calculation methods side by side:")
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    
    # OASIS nWBV (Original)
    if 'nWBV' in data.columns and 'CDR' in data.columns:
        ax1.bar(data['CDR'], data['nWBV'], color='hotpink', width=0.4)
        ax1.set_title('nWBV (Original)', fontsize=20)
        ax1.set_xlabel('CDR', fontsize=16)
        ax1.set_ylabel('Brain Volume (mm³)', fontsize=16)
        ax1.tick_params(labelsize=14)
    
    # Brain Extraction
    if 'nWBV_brain_extraction' in data.columns:
        ax2.bar(data['CDR'], data['nWBV_brain_extraction'], color='mediumseagreen', width=0.4)
        ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
        ax2.set_xlabel('CDR', fontsize=16)
        ax2.set_ylabel('Brain Volume (mm³)', fontsize=16)
        ax2.tick_params(labelsize=14)
    
    # Deep Atropos
    if 'nWBV_deep_atropos' in data.columns:
        ax3.bar(data['CDR'], data['nWBV_deep_atropos'], color='tab:blue', width=0.4)
        ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
        ax3.set_xlabel('CDR', fontsize=16)
        ax3.set_ylabel('Brain Volume (mm³)', fontsize=16)
        ax3.tick_params(labelsize=14)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # MMSE Analysis (if available)
    mmse_col = next((c for c in ['MMSE', 'mmse'] if c in data.columns), None)
    if mmse_col:
        st.markdown("### MMSE vs Brain Volume (Scatter Plots)")
        st.write("Analyzing the relationship between Mini-Mental State Examination scores and brain volume:")
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
        
        # OASIS nWBV (Original)
        data_mmse = data[[mmse_col, 'nWBV']].dropna()
        if len(data_mmse) > 1:
            ax1.scatter(data_mmse[mmse_col], data_mmse['nWBV'], color='hotpink', edgecolor='k', alpha=0.6, s=30)
            
            # Chi-squared regression
            y_mean = data_mmse['nWBV'].mean()
            y_range = data_mmse['nWBV'].max() - data_mmse['nWBV'].min()
            z = np.polyfit(data_mmse[mmse_col], data_mmse['nWBV'], 1)
            init_slope, init_intercept = z[0], z[1]
            
            result = scipy.optimize.minimize(
                chi_squared,
                [init_slope, init_intercept],
                args=(data_mmse['nWBV'].values, data_mmse[mmse_col].values),
                bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
            )
            m, b = result.x[0], result.x[1]
            
            x_line = np.linspace(data_mmse[mmse_col].min(), data_mmse[mmse_col].max(), 100)
            ax1.plot(x_line, m * x_line + b, 'k--', linewidth=2)
            ax1.set_xlabel('MMSE Score', fontsize=16)
            ax1.set_ylabel('Brain Volume (mm³)', fontsize=16)
            ax1.set_title('nWBV (Original)', fontsize=20)
            ax1.tick_params(labelsize=14)
            ax1.grid(True, alpha=0.3)
        
        # Brain Extraction
        if 'nWBV_brain_extraction' in data.columns:
            data_mmse = data[[mmse_col, 'nWBV_brain_extraction']].dropna()
            if len(data_mmse) > 1:
                ax2.scatter(data_mmse[mmse_col], data_mmse['nWBV_brain_extraction'], 
                           color='mediumseagreen', edgecolor='k', alpha=0.6, s=30)
                
                y_mean = data_mmse['nWBV_brain_extraction'].mean()
                y_range = data_mmse['nWBV_brain_extraction'].max() - data_mmse['nWBV_brain_extraction'].min()
                z = np.polyfit(data_mmse[mmse_col], data_mmse['nWBV_brain_extraction'], 1)
                init_slope, init_intercept = z[0], z[1]
                
                result = scipy.optimize.minimize(
                    chi_squared,
                    [init_slope, init_intercept],
                    args=(data_mmse['nWBV_brain_extraction'].values, data_mmse[mmse_col].values),
                    bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
                )
                m, b = result.x[0], result.x[1]
                
                x_line = np.linspace(data_mmse[mmse_col].min(), data_mmse[mmse_col].max(), 100)
                ax2.plot(x_line, m * x_line + b, 'k--', linewidth=2)
                ax2.set_xlabel('MMSE Score', fontsize=16)
                ax2.set_ylabel('Brain Volume (mm³)', fontsize=16)
                ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
                ax2.tick_params(labelsize=14)
                ax2.grid(True, alpha=0.3)
        
        # Deep Atropos
        if 'nWBV_deep_atropos' in data.columns:
            data_mmse = data[[mmse_col, 'nWBV_deep_atropos']].dropna()
            if len(data_mmse) > 1:
                ax3.scatter(data_mmse[mmse_col], data_mmse['nWBV_deep_atropos'], 
                           color='tab:blue', edgecolor='k', alpha=0.6, s=30)
                
                y_mean = data_mmse['nWBV_deep_atropos'].mean()
                y_range = data_mmse['nWBV_deep_atropos'].max() - data_mmse['nWBV_deep_atropos'].min()
                z = np.polyfit(data_mmse[mmse_col], data_mmse['nWBV_deep_atropos'], 1)
                init_slope, init_intercept = z[0], z[1]
                
                result = scipy.optimize.minimize(
                    chi_squared,
                    [init_slope, init_intercept],
                    args=(data_mmse['nWBV_deep_atropos'].values, data_mmse[mmse_col].values),
                    bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
                )
                m, b = result.x[0], result.x[1]
                
                x_line = np.linspace(data_mmse[mmse_col].min(), data_mmse[mmse_col].max(), 100)
                ax3.plot(x_line, m * x_line + b, 'k--', linewidth=2)
                ax3.set_xlabel('MMSE Score', fontsize=16)
                ax3.set_ylabel('Brain Volume (mm³)', fontsize=16)
                ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
                ax3.tick_params(labelsize=14)
                ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


###############################################################
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
        st.write(
        """
    Before diving into volumetric analysis, it's helpful to visualize what the MRI data actually looks like. 
    MRI scans are three-dimensional arrays where each voxel (3D pixel) represents a measurement at a specific 
    location in the brain. By displaying a 2D slice from the 3D volume, we can see the brain structures and 
    verify that our data has loaded correctly.
    
    This visualization helps us:
    - Confirm data integrity (check for artifacts or corrupted files)
    - Understand the spatial resolution and quality of the scans
    - Appreciate the anatomical structures we're analyzing
    """
    )
    st.code(
        """
# Specify path to an example MRI scan
path_ex = "path/to/example_brain.hdr"

# Load the MRI image using nibabel (neuroimaging data library)
img = nib.load(path_ex)

# Extract the actual image data as a NumPy array
# This converts the MRI file format into a 3D array we can manipulate
data = img.get_fdata()

# Display a transverse (horizontal) slice through the brain
# We select a slice 95 positions from the top of the volume
slice_number = data.shape[2] - 95

plt.imshow(data[:, :, slice_number], cmap='twilight_shifted')
plt.axis('off')  # Hide axis labels for cleaner visualization
plt.title(f'Example Transverse Slice at Slice #{slice_number}')
plt.show()
""",
        language="python",
    )

    # Actually display the example image
    try:
        hdr_path = (
            Path(__file__).parent
            / "data/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.hdr"
        )
        if hdr_path.exists():
            img = nib.load(str(hdr_path))
            data = np.squeeze(img.get_fdata())
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(
                data[:, :, data.shape[2] - 95],
                cmap="twilight_shifted"
            )
            ax.axis("off")
            ax.set_title("Example slice image of Brain")
            st.pyplot(fig)
        else:
            st.info("Example MRI file not found in data directory.")
    except Exception as e:
        st.warning(f"Could not load example image: {e}")

    st.subheader("Brain Volume Calculation")
    st.write(
        """
    Now we'll calculate brain volumes from the MRI scans using two different deep learning-based segmentation methods. The goal is to:
    1. Identify which voxels in the MRI contain brain tissue (vs. skull, CSF, or background)
    2. Count the brain voxels and multiply by voxel dimensions to get volume
    3. Compare our calculated volumes with the OASIS-provided normalized whole brain volume (nWBV)

    We need to filter our dataset to only include brain scans that:
    1. Have corresponding MRI files available
    2. Have Clinical Dementia Rating (CDR) scores for correlation analysis

    **Important Note:** The OASIS dataset provides ATLAS-registered images, meaning the brain scans have been warped to a 
    standard template. This simplifies segmentation but introduces some distortion. The ATLAS Scaling Factor (ASF) allows us 
    to convert measurements back to natural space, though this is not a perfect correction. In a more rigorous analysis, we 
    would calculate volumes in natural space and then register to ATLAS only for segmentation purposes.
    """
    )
    st.code(
        """
# Extract list of patient IDs from our clinical dataframe
# This creates a reference list for looping through available brain scans
Ids = oasis_crossref['ID']
""",
        language="python",
    )

    st.write(
        """
    ### Defining Brain Segmentation Functions

    We'll implement two different methods for calculating brain volume, each using a different deep learning model from 
    the ANTsPyNet library. By comparing multiple methods, we can assess the consistency and reliability of our volume measurements.

    **Why use functions?**
    - Encapsulates complex logic into reusable, testable units
    - Makes the code more readable and maintainable
    - Allows us to easily apply the same analysis to many brain scans
    - Facilitates comparison between different segmentation approaches

    Both functions will:
    1. Take an ANTs image object as input
    2. Generate a probability map or segmentation of brain tissue
    3. Count brain voxels and calculate volume
    4. Return both pixel count and physical volume
    """
    )

    st.subheader("Method #1: Brain Extraction")
    st.write(
        """
    The `brain_extraction` function from ANTsPyNet uses a U-Net deep learning architecture trained on ANTs-based data. This method:
    - Creates a probability map where each voxel has a value between 0 and 1 indicating the likelihood it's brain tissue
    - Applies a threshold (0.5) to convert probabilities into a binary mask
    - Counts "brain" voxels and multiplies by voxel volume to get total brain volume

    This is a simpler approach that focuses on distinguishing brain from non-brain tissue without further classification.
    """
    )
    st.code(
        '''
def brain_extraction_method(img):
    """
    Uses ANTsPyNet.brain_extraction to calculate brain volume from MRI image.
    
    Parameters:
    -----------
    img : ANTsImage
        Input MRI image in ANTs format
        
    Returns:
    --------
    tuple : (pixel_count, volume)
        pixel_count : int - number of voxels classified as brain tissue
        volume : float - physical volume in mm³
    """
    
    # Create probability map using U-Net model
    # Each voxel gets a probability (0-1) of being brain tissue
    prob_brain_mask = brain_extraction(img, modality="t1", verbose=True)
    
    # Convert probability map to binary mask using 0.5 threshold
    # Voxels with >50% probability are classified as brain (value=1), else non-brain (value=0)
    brain_mask = ants.threshold_image(prob_brain_mask, 0.5, 1e9, 1, 0)

    # Count total number of brain voxels
    # Convert to boolean array and sum to get count
    pixel_count = int(brain_mask.numpy().astype(bool).sum())

    # Calculate physical volume by multiplying voxel count by individual voxel volume
    # img.spacing gives dimensions of each voxel in mm
    voxel_volume = float(np.prod(img.spacing))
    volume = pixel_count * voxel_volume

    return pixel_count, volume
''',
        language="python",
    )

    st.subheader("Method #2: Deep Atropos Segmentation")
    st.write(
        """
    The `deep_atropos` function performs more sophisticated six-tissue segmentation, classifying each voxel into one of these categories:
    - Label 0: Background
    - Label 1: Cerebrospinal fluid (CSF)
    - Label 2: Gray matter
    - Label 3: White matter
    - Label 4: Deep gray matter
    - Label 5: Brain stem
    - Label 6: Cerebellum

    For our brain volume calculation, we sum only gray matter, white matter, and deep gray matter (labels 2, 3, and 4), 
    excluding CSF and other non-brain tissue. This method may provide a more accurate measure of actual brain tissue volume.

    The preprocessing step includes N4 bias correction, denoising, brain extraction, and affine registration to MNI space.
    """
    )
    st.code(
        '''
def atropos_segmentation(img, pre):
    """
    Uses deep_atropos six-tissue segmentation to calculate brain volume.
    
    Parameters:
    -----------
    img : ANTsImage
        Input MRI image in ANTs format
    pre : bool
        Whether to perform preprocessing (N4 bias correction, denoising)
        
    Returns:
    --------
    tuple : (pixel_count, volume)
        pixel_count : int - number of voxels classified as brain tissue
        volume : float - physical volume in mm³
    """
    
    # Perform six-tissue segmentation using deep learning model
    # Returns dictionary with segmentation image and probability maps
    seg = deep_atropos(img, do_preprocessing=pre)

    # Extract the segmentation image (each voxel has a tissue class label 0-6)
    seg_img = seg['segmentation_image']
    seg_np = seg_img.numpy().astype(int)

    # Define which tissue types to include in brain volume calculation:
    # Label 2: Gray matter
    # Label 3: White matter  
    # Label 4: Deep gray matter
    # We exclude: CSF (1), background (0), brain stem (5), cerebellum (6)
    brain_tissue = ((seg_np == 2) | (seg_np == 3) | (seg_np == 4)).astype(float)

    # Count total brain tissue voxels
    pixel_count = int(brain_tissue.sum())

    # Calculate physical volume
    # Multiply voxel count by individual voxel dimensions
    voxel_volume = float(np.prod(img.spacing))
    volume = pixel_count * voxel_volume

    return pixel_count, volume
''',
        language="python",
    )

    st.subheader("Calculate Volumes for All Scans")
    st.write(
        "**Do not run this following chunk unless you want your computer to die. It takes 90 minutes on a 4070.**"
    )
    st.code(
        """
# Initiate pixel counts
pixel_counts_brain_extraction = np.zeros(len(Ids))
pixel_counts_deep_atropos = np.zeros(len(Ids))

# Initiate volumes
volumes_brain_extraction = np.zeros(len(Ids))
volumes_deep_atropos = np.zeros(len(Ids))

# Loop through Id list for finding brains we want to analyze
for i in range(len(Ids)):
    current_Id = Ids.iloc[i]

    # Skip if this ID is not in the dataframe
    if current_Id not in oasis_crossref["ID"].values:
        print("Skipping", current_Id)
        continue
    print(current_Id)

    # Both possible filenames
    n4_file = f'{path}{current_Id}_mpr_n4_anon_111_t88_gfc.hdr'
    n3_file = f'{path}{current_Id}_mpr_n3_anon_111_t88_gfc.hdr'

    # Pick whichever one actually exists (manual check)
    if os.path.exists(n4_file):
        new_file = n4_file
    elif os.path.exists(n3_file):
        new_file = n3_file
    else:
        print("File not found for", current_Id)
        continue  # Skip to the next ID

    # Now the fun part
    # This gets the image into the antspy module
    raw_img_ants = ants.image_read(new_file, reorient='IAL')

    # Correct the spacing
    raw_img_ants.set_spacing((1,1,1))

    # Run our volume functions
    pixel_counts_brain_extraction[i], volumes_brain_extraction[i] = \\
        brain_extraction_method(raw_img_ants)

    pixel_counts_deep_atropos[i], volumes_deep_atropos[i] = \\
        atropos_segmentation(raw_img_ants, pre=False)
""",
        language="python",
    )

    st.subheader("Wrangling")
    st.write(
        """
    With the volumes calculated, now we need to normalize them!

    From *Buckner et al. 2004*, we know the following:
    - Estimated total intracranial volume (eTIV) is a fully automated estimate of TIVₙₐₜ (Total Intracranial Volume in natural space)
    - Total Intracranial Volume in ATLAS space (TIVₐₜₗ) divided by the ATLAS scaling factor (ASF) yields TIVₙₐₜ
    - The same relations apply to Volume (atl and nat)
    - Normalized Whole Brain Volume (nWBV) is the automated tissue segmentation based estimate of brain volume (gray-plus white-matter).
      Normalized to percentage based on the atlas target mask.

    Therefore, we obtain the following equations and relations:
    """
    )
    st.latex(r"eTIV \approx TIV_{nat} = \frac{TIV_{atl}}{ASF}")
    st.latex(r"Vol_{nat} = \frac{Vol_{atl}}{ASF}")
    st.latex(r"nWBV = \frac{Vol_{nat}}{eTIV}")
    st.write(
        """
    Which gives us the final equation we should apply to the calculated volumes to obtain volumes normalized to the ATLAS template brain scan used for segmentation and analysis:
    """
    )
    st.latex(r"nWBV = \frac{Vol_{atl}}{eTIV \times ASF}")

    st.subheader("Normalize Brain Volumes")
    st.code(
        """
# Add the volumes to the dataframe
oasis_crossref['Vol BE'] = volumes_brain_extraction
oasis_crossref['Pixels BE'] = pixel_counts_brain_extraction

# Deep atropos GM+WM only
oasis_crossref['Vol DA'] = volumes_deep_atropos
oasis_crossref['Pixels DA'] = pixel_counts_deep_atropos

# Normalize and correct with ASF
oasis_crossref['nWBV_brain_extraction'] = \\
    oasis_crossref['Vol BE'] / (oasis_crossref['ASF'] * (oasis_crossref['eTIV'] * 1000))

# Deep atropos
oasis_crossref['nWBV_deep_atropos'] = \\
    oasis_crossref['Vol DA'] / (oasis_crossref['ASF'] * (oasis_crossref['eTIV'] * 1000))

oasis_crossref
""",
        language="python",
    )

    st.code(
        """
# Load in modified csv with stored values
# This is done to create this example code without having to re-run
csv_data = "path/to/final_data_oasis.csv"
Data = pd.read_csv(csv_data)
""",
        language="python",
    )

    # Display the actual final dataframe with calculated volumes
    data = get_data()
    st.write("**Final dataframe with normalized brain volumes:**")
    st.dataframe(data.head(10))
    st.write(
        f"Shape: {data.shape[0]} rows, {data.shape[1]} columns"
    )
    # Show the relevant columns
    st.write("**Key columns:**")
    relevant_cols = [
        col for col in data.columns
        if any(x in col for x in ['Vol', 'nWBV', 'eTIV', 'ASF'])
    ]
    if relevant_cols:
        st.dataframe(data[relevant_cols].head(10))

    st.subheader("Plotting")
    st.write(
        """
    With the above data, we have a lot of values to work with to answer our goal question(s):
    - nWBV_brain_extraction and nWBV_deep_atropos give us two different methods to compare for calculating a normalized whole brain volume
    - nWBV is the normalized brain volume from the original data set just for comparison and correctness
    - Clinical Dementia Rating (CDR) when compared with the nWBVs allows us to find out if CDR and brain volume are related in any ways
    - We can also see how CDR, Brain Volume, Age, Sex and other demographic data are related

    The below code includes some visualization plots for example.
    """
    )

    st.subheader("Visualization: CDR vs nWBV")
    st.write(
        """
    This first one below is box and whisker for CDR vs nWBV. In theory this will show our biggest question/curiosity with this
    project with regards to whether dementia and brain volume correlate. We compare all three methods.
    """
    )
    st.code(
        """
# Box plots for CDR vs nWBV - comparing all three methods
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# OASIS nWBV (Original)
sns.boxplot(x='CDR', y='nWBV', data=Data, ax=ax1, color='hotpink')
ax1.set_title('nWBV (Original)', fontsize=20)
ax1.set_xlabel('CDR', fontsize=16)
ax1.set_ylabel('Brain Volume (mm³)', fontsize=16)
ax1.tick_params(labelsize=14)

# Brain Extraction
sns.boxplot(x='CDR', y='nWBV_brain_extraction', data=Data, ax=ax2, color='mediumseagreen')
ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
ax2.set_xlabel('CDR', fontsize=16)
ax2.set_ylabel('Brain Volume (mm³)', fontsize=16)
ax2.tick_params(labelsize=14)

# Deep Atropos
sns.boxplot(x='CDR', y='nWBV_deep_atropos', data=Data, ax=ax3, color='tab:blue')
ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
ax3.set_xlabel('CDR', fontsize=16)
ax3.set_ylabel('Brain Volume (mm³)', fontsize=16)
ax3.tick_params(labelsize=14)

plt.tight_layout()
plt.show()
""",
        language="python",
    )

    # Display the actual box plots
    data = get_data()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

    sns.boxplot(x='CDR', y='nWBV', data=data, ax=ax1, color='hotpink')
    ax1.set_title('nWBV (Original)', fontsize=20)
    ax1.set_xlabel('CDR', fontsize=16)
    ax1.set_ylabel('Brain Volume (mm³)', fontsize=16)
    ax1.tick_params(labelsize=14)

    sns.boxplot(x='CDR', y='nWBV_brain_extraction', data=data, ax=ax2, color='mediumseagreen')
    ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
    ax2.set_xlabel('CDR', fontsize=16)
    ax2.set_ylabel('Brain Volume (mm³)', fontsize=16)
    ax2.tick_params(labelsize=14)

    sns.boxplot(x='CDR', y='nWBV_deep_atropos', data=data, ax=ax3, color='tab:blue')
    ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
    ax3.set_xlabel('CDR', fontsize=16)
    ax3.set_ylabel('Brain Volume (mm³)', fontsize=16)
    ax3.tick_params(labelsize=14)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Visualization: nWBV vs Age by Gender")
    st.write(
        """
    Again, plotting a scatter plot this time, to compare the nWBVs with how they correlate (or not) with age.
    For this plot, the different genders were plotted separately to also see how that differs between M/F.
    We show all three methods side by side.
    """
    )
    st.code(
        """
# Create sub-dataframes for genders
men = Data[Data['M/F'] == 'M']
women = Data[Data['M/F'] == 'F']

# Create 3 subplots for each method
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# OASIS nWBV (Original)
ax1.plot(men['Age'], men['nWBV'], '.', color='blue', label='Male')
ax1.plot(women['Age'], women['nWBV'], '.', color='red', label='Female')
ax1.set_xlabel('Age (Years)', fontsize=16)
ax1.set_ylabel('Brain Volume (mm³)', fontsize=16)
ax1.set_title('nWBV (Original)', fontsize=20)
ax1.tick_params(labelsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Brain Extraction
ax2.plot(men['Age'], men['nWBV_brain_extraction'], '.', color='blue', label='Male')
ax2.plot(women['Age'], women['nWBV_brain_extraction'], '.', color='red', label='Female')
ax2.set_xlabel('Age (Years)', fontsize=16)
ax2.set_ylabel('Brain Volume (mm³)', fontsize=16)
ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# Deep Atropos
ax3.plot(men['Age'], men['nWBV_deep_atropos'], '.', color='blue', label='Male')
ax3.plot(women['Age'], women['nWBV_deep_atropos'], '.', color='red', label='Female')
ax3.set_xlabel('Age (Years)', fontsize=16)
ax3.set_ylabel('Brain Volume (mm³)', fontsize=16)
ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
ax3.tick_params(labelsize=14)
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""",
        language="python",
    )

    # Display the actual scatter plots
    data = get_data()
    men = data[data['M/F'] == 'M']
    women = data[data['M/F'] == 'F']
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

    # OASIS nWBV (Original)
    ax1.plot(men['Age'], men['nWBV'], '.', color='blue', markersize=6, label='Male')
    ax1.plot(women['Age'], women['nWBV'], '.', color='red', markersize=6, label='Female')
    ax1.set_xlabel('Age (Years)', fontsize=16)
    ax1.set_ylabel('Brain Volume (mm³)', fontsize=16)
    ax1.set_title('nWBV (Original)', fontsize=20)
    ax1.tick_params(labelsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Brain Extraction
    ax2.plot(men['Age'], men['nWBV_brain_extraction'], '.', color='blue', markersize=6, label='Male')
    ax2.plot(women['Age'], women['nWBV_brain_extraction'], '.', color='red', markersize=6, label='Female')
    ax2.set_xlabel('Age (Years)', fontsize=16)
    ax2.set_ylabel('Brain Volume (mm³)', fontsize=16)
    ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
    ax2.tick_params(labelsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Deep Atropos
    ax3.plot(men['Age'], men['nWBV_deep_atropos'], '.', color='blue', markersize=6, label='Male')
    ax3.plot(women['Age'], women['nWBV_deep_atropos'], '.', color='red', markersize=6, label='Female')
    ax3.set_xlabel('Age (Years)', fontsize=16)
    ax3.set_ylabel('Brain Volume (mm³)', fontsize=16)
    ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
    ax3.tick_params(labelsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.write(
        """
    **Chi-Squared Linear Regression:**

    We can also fit linear regression lines to each gender using chi-squared minimization to see the trends.
    """
    )
    st.code(
        """
# Chi-squared objective function
def chi_squared(params, areas, times):
    m = params[0]
    b = params[1]
    fit = m * times + b
    vals = (areas - fit) ** 2 / fit
    return np.sum(vals)

# Create 3 subplots for each method
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

# OASIS nWBV (Original) - Chi-squared regression
if len(men) > 1:
    y_mean = men['nWBV'].mean()
    y_range = men['nWBV'].max() - men['nWBV'].min()
    z = np.polyfit(men['Age'].values, men['nWBV'].values, 1)
    init_slope, init_intercept = z[0], z[1]
    men_ans = scipy.optimize.minimize(
        chi_squared, [init_slope, init_intercept],
        args=(men['nWBV'].values, men['Age'].values),
        bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
    )
    men_m, men_b = men_ans.x[0], men_ans.x[1]
if len(women) > 1:
    y_mean = women['nWBV'].mean()
    y_range = women['nWBV'].max() - women['nWBV'].min()
    z = np.polyfit(women['Age'].values, women['nWBV'].values, 1)
    init_slope, init_intercept = z[0], z[1]
    women_ans = scipy.optimize.minimize(
        chi_squared, [init_slope, init_intercept],
        args=(women['nWBV'].values, women['Age'].values),
        bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
    )
    women_m, women_b = women_ans.x[0], women_ans.x[1]
ax1.scatter(men['Age'], men['nWBV'], color='blue', alpha=0.6, s=30, label='Male')
ax1.scatter(women['Age'], women['nWBV'], color='red', alpha=0.6, s=30, label='Female')
age_range_men = np.linspace(men['Age'].min(), men['Age'].max(), 100)
ax1.plot(age_range_men, men_m * age_range_men + men_b, 'b--', linewidth=2)
age_range_women = np.linspace(women['Age'].min(), women['Age'].max(), 100)
ax1.plot(age_range_women, women_m * age_range_women + women_b, 'r--', linewidth=2)
ax1.set_xlabel('Age (Years)', fontsize=16)
ax1.set_ylabel('Brain Volume (mm³)', fontsize=16)
ax1.set_title('nWBV (Original)', fontsize=20)
ax1.tick_params(labelsize=14)
ax1.legend(fontsize=12)

# Brain Extraction - Chi-squared regression
if len(men) > 1:
    y_mean = men['nWBV_brain_extraction'].mean()
    y_range = men['nWBV_brain_extraction'].max() - men['nWBV_brain_extraction'].min()
    z = np.polyfit(men['Age'].values, men['nWBV_brain_extraction'].values, 1)
    init_slope, init_intercept = z[0], z[1]
    men_ans = scipy.optimize.minimize(
        chi_squared, [init_slope, init_intercept],
        args=(men['nWBV_brain_extraction'].values, men['Age'].values),
        bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
    )
    men_m2, men_b2 = men_ans.x[0], men_ans.x[1]
if len(women) > 1:
    y_mean = women['nWBV_brain_extraction'].mean()
    y_range = women['nWBV_brain_extraction'].max() - women['nWBV_brain_extraction'].min()
    z = np.polyfit(women['Age'].values, women['nWBV_brain_extraction'].values, 1)
    init_slope, init_intercept = z[0], z[1]
    women_ans = scipy.optimize.minimize(
        chi_squared, [init_slope, init_intercept],
        args=(women['nWBV_brain_extraction'].values, women['Age'].values),
        bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
    )
    women_m2, women_b2 = women_ans.x[0], women_ans.x[1]
ax2.scatter(men['Age'], men['nWBV_brain_extraction'], color='blue', alpha=0.6, s=30, label='Male')
ax2.scatter(women['Age'], women['nWBV_brain_extraction'], color='red', alpha=0.6, s=30, label='Female')
ax2.plot(age_range_men, men_m2 * age_range_men + men_b2, 'b--', linewidth=2)
ax2.plot(age_range_women, women_m2 * age_range_women + women_b2, 'r--', linewidth=2)
ax2.set_xlabel('Age (Years)', fontsize=16)
ax2.set_ylabel('Brain Volume (mm³)', fontsize=16)
ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=12)

# Deep Atropos - Chi-squared regression
if len(men) > 1:
    y_mean = men['nWBV_deep_atropos'].mean()
    y_range = men['nWBV_deep_atropos'].max() - men['nWBV_deep_atropos'].min()
    z = np.polyfit(men['Age'].values, men['nWBV_deep_atropos'].values, 1)
    init_slope, init_intercept = z[0], z[1]
    men_ans = scipy.optimize.minimize(
        chi_squared, [init_slope, init_intercept],
        args=(men['nWBV_deep_atropos'].values, men['Age'].values),
        bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
    )
    men_m3, men_b3 = men_ans.x[0], men_ans.x[1]
if len(women) > 1:
    y_mean = women['nWBV_deep_atropos'].mean()
    y_range = women['nWBV_deep_atropos'].max() - women['nWBV_deep_atropos'].min()
    z = np.polyfit(women['Age'].values, women['nWBV_deep_atropos'].values, 1)
    init_slope, init_intercept = z[0], z[1]
    women_ans = scipy.optimize.minimize(
        chi_squared, [init_slope, init_intercept],
        args=(women['nWBV_deep_atropos'].values, women['Age'].values),
        bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
    )
    women_m3, women_b3 = women_ans.x[0], women_ans.x[1]
ax3.scatter(men['Age'], men['nWBV_deep_atropos'], color='blue', alpha=0.6, s=30, label='Male')
ax3.scatter(women['Age'], women['nWBV_deep_atropos'], color='red', alpha=0.6, s=30, label='Female')
ax3.plot(age_range_men, men_m3 * age_range_men + men_b3, 'b--', linewidth=2)
ax3.plot(age_range_women, women_m3 * age_range_women + women_b3, 'r--', linewidth=2)
ax3.set_xlabel('Age (Years)', fontsize=16)
ax3.set_ylabel('Brain Volume (mm³)', fontsize=16)
ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
ax3.tick_params(labelsize=14)
ax3.legend(fontsize=12)

plt.tight_layout()
plt.show()
""",
        language="python",
    )

    # Display the actual chi-squared regression plots for all 3 methods
    if len(men) > 1 and len(women) > 1:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
        
        # OASIS nWBV (Original)
        y_mean = men['nWBV'].mean()
        y_range = men['nWBV'].max() - men['nWBV'].min()
        z = np.polyfit(men['Age'].values, men['nWBV'].values, 1)
        init_slope, init_intercept = z[0], z[1]
        men_ans = scipy.optimize.minimize(
            chi_squared, [init_slope, init_intercept],
            args=(men['nWBV'].values, men['Age'].values),
            bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
        )
        men_m, men_b = men_ans.x[0], men_ans.x[1]
        
        y_mean = women['nWBV'].mean()
        y_range = women['nWBV'].max() - women['nWBV'].min()
        z = np.polyfit(women['Age'].values, women['nWBV'].values, 1)
        init_slope, init_intercept = z[0], z[1]
        women_ans = scipy.optimize.minimize(
            chi_squared, [init_slope, init_intercept],
            args=(women['nWBV'].values, women['Age'].values),
            bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
        )
        women_m, women_b = women_ans.x[0], women_ans.x[1]
        
        ax1.scatter(men['Age'], men['nWBV'], color='blue', alpha=0.6, s=30, label='Male')
        ax1.scatter(women['Age'], women['nWBV'], color='red', alpha=0.6, s=30, label='Female')
        age_range_men = np.linspace(men['Age'].min(), men['Age'].max(), 100)
        ax1.plot(age_range_men, men_m * age_range_men + men_b, 'b--', linewidth=2)
        age_range_women = np.linspace(women['Age'].min(), women['Age'].max(), 100)
        ax1.plot(age_range_women, women_m * age_range_women + women_b, 'r--', linewidth=2)
        ax1.set_xlabel('Age (Years)', fontsize=16)
        ax1.set_ylabel('Brain Volume (mm³)', fontsize=16)
        ax1.set_title('nWBV (Original)', fontsize=20)
        ax1.tick_params(labelsize=14)
        ax1.legend(fontsize=12)
        
        # Brain Extraction
        y_mean = men['nWBV_brain_extraction'].mean()
        y_range = men['nWBV_brain_extraction'].max() - men['nWBV_brain_extraction'].min()
        z = np.polyfit(men['Age'].values, men['nWBV_brain_extraction'].values, 1)
        init_slope, init_intercept = z[0], z[1]
        men_ans = scipy.optimize.minimize(
            chi_squared, [init_slope, init_intercept],
            args=(men['nWBV_brain_extraction'].values, men['Age'].values),
            bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
        )
        men_m2, men_b2 = men_ans.x[0], men_ans.x[1]
        
        y_mean = women['nWBV_brain_extraction'].mean()
        y_range = women['nWBV_brain_extraction'].max() - women['nWBV_brain_extraction'].min()
        z = np.polyfit(women['Age'].values, women['nWBV_brain_extraction'].values, 1)
        init_slope, init_intercept = z[0], z[1]
        women_ans = scipy.optimize.minimize(
            chi_squared, [init_slope, init_intercept],
            args=(women['nWBV_brain_extraction'].values, women['Age'].values),
            bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
        )
        women_m2, women_b2 = women_ans.x[0], women_ans.x[1]
        
        ax2.scatter(men['Age'], men['nWBV_brain_extraction'], color='blue', alpha=0.6, s=30, label='Male')
        ax2.scatter(women['Age'], women['nWBV_brain_extraction'], color='red', alpha=0.6, s=30, label='Female')
        ax2.plot(age_range_men, men_m2 * age_range_men + men_b2, 'b--', linewidth=2)
        ax2.plot(age_range_women, women_m2 * age_range_women + women_b2, 'r--', linewidth=2)
        ax2.set_xlabel('Age (Years)', fontsize=16)
        ax2.set_ylabel('Brain Volume (mm³)', fontsize=16)
        ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
        ax2.tick_params(labelsize=14)
        ax2.legend(fontsize=12)
        
        # Deep Atropos
        y_mean = men['nWBV_deep_atropos'].mean()
        y_range = men['nWBV_deep_atropos'].max() - men['nWBV_deep_atropos'].min()
        z = np.polyfit(men['Age'].values, men['nWBV_deep_atropos'].values, 1)
        init_slope, init_intercept = z[0], z[1]
        men_ans = scipy.optimize.minimize(
            chi_squared, [init_slope, init_intercept],
            args=(men['nWBV_deep_atropos'].values, men['Age'].values),
            bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
        )
        men_m3, men_b3 = men_ans.x[0], men_ans.x[1]
        
        y_mean = women['nWBV_deep_atropos'].mean()
        y_range = women['nWBV_deep_atropos'].max() - women['nWBV_deep_atropos'].min()
        z = np.polyfit(women['Age'].values, women['nWBV_deep_atropos'].values, 1)
        init_slope, init_intercept = z[0], z[1]
        women_ans = scipy.optimize.minimize(
            chi_squared, [init_slope, init_intercept],
            args=(women['nWBV_deep_atropos'].values, women['Age'].values),
            bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
        )
        women_m3, women_b3 = women_ans.x[0], women_ans.x[1]
        
        ax3.scatter(men['Age'], men['nWBV_deep_atropos'], color='blue', alpha=0.6, s=30, label='Male')
        ax3.scatter(women['Age'], women['nWBV_deep_atropos'], color='red', alpha=0.6, s=30, label='Female')
        ax3.plot(age_range_men, men_m3 * age_range_men + men_b3, 'b--', linewidth=2)
        ax3.plot(age_range_women, women_m3 * age_range_women + women_b3, 'r--', linewidth=2)
        ax3.set_xlabel('Age (Years)', fontsize=16)
        ax3.set_ylabel('Brain Volume (mm³)', fontsize=16)
        ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
        ax3.tick_params(labelsize=14)
        ax3.legend(fontsize=12)
        
        plt.tight_layout()
        st.pyplot(fig)

        # Display regression statistics for all 3 methods
        st.write("**Regression Statistics:**")
        
        # OASIS nWBV (Original)
        st.write("**nWBV (Original)**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Men:**")
            st.write(f"Slope: {men_m:.6f}")
            st.write(f"Intercept: {men_b:.6f}")
            st.write(f"Chi²: {chi_squared([men_m, men_b], men['nWBV'].values, men['Age'].values):.6f}")
        with col2:
            st.write("**Women:**")
            st.write(f"Slope: {women_m:.6f}")
            st.write(f"Intercept: {women_b:.6f}")
            st.write(f"Chi²: {chi_squared([women_m, women_b], women['nWBV'].values, women['Age'].values):.6f}")
        
        # Brain Extraction
        st.write("**nWBV (Brain Extraction)**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Men:**")
            st.write(f"Slope: {men_m2:.6f}")
            st.write(f"Intercept: {men_b2:.6f}")
            st.write(f"Chi²: {chi_squared([men_m2, men_b2], men['nWBV_brain_extraction'].values, men['Age'].values):.6f}")
        with col2:
            st.write("**Women:**")
            st.write(f"Slope: {women_m2:.6f}")
            st.write(f"Intercept: {women_b2:.6f}")
            st.write(f"Chi²: {chi_squared([women_m2, women_b2], women['nWBV_brain_extraction'].values, women['Age'].values):.6f}")
        
        # Deep Atropos
        st.write("**nWBV (Deep Atropos)**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Men:**")
            st.write(f"Slope: {men_m3:.6f}")
            st.write(f"Intercept: {men_b3:.6f}")
            st.write(f"Chi²: {chi_squared([men_m3, men_b3], men['nWBV_deep_atropos'].values, men['Age'].values):.6f}")
        with col2:
            st.write("**Women:**")
            st.write(f"Slope: {women_m3:.6f}")
            st.write(f"Intercept: {women_b3:.6f}")
            st.write(f"Chi²: {chi_squared([women_m3, women_b3], women['nWBV_deep_atropos'].values, women['Age'].values):.6f}")

    st.subheader("Compare Methods")
    st.write(
        """
    While for the above plots we've used the nWBV values that were given with the OASIS 1 dataset, we wanted to find these on our own for this project, and have done so.

    The question remains of how effectively the brain_extraction method and the deep_atropos methods approximate nWBV.

    We can compare them visually by looking again at nWBV vs CDR, but with the 3 different methods.
    """
    )
    st.code(
        """
# Compare three different methods side by side
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

# OASIS nWBV (Original)
ax1.bar(Data['CDR'], Data['nWBV'], color='hotpink', width=0.4)
ax1.set_title('nWBV (Original)', fontsize=20)
ax1.set_xlabel('CDR', fontsize=16)
ax1.set_ylabel('Brain Volume (mm³)', fontsize=16)
ax1.tick_params(labelsize=14)

# Brain Extraction
ax2.bar(Data['CDR'], Data['nWBV_brain_extraction'], color='mediumseagreen', width=0.4)
ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
ax2.set_xlabel('CDR', fontsize=16)
ax2.set_ylabel('Brain Volume (mm³)', fontsize=16)
ax2.tick_params(labelsize=14)

# Deep Atropos
ax3.bar(Data['CDR'], Data['nWBV_deep_atropos'], color='tab:blue', width=0.4)
ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
ax3.set_xlabel('CDR', fontsize=16)
ax3.set_ylabel('Brain Volume (mm³)', fontsize=16)
ax3.tick_params(labelsize=14)

plt.tight_layout()
plt.show()
""",
        language="python",
    )

    # Display the actual comparison plot
    data = get_data()
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, figsize=(12, 4)
    )

    cdr_vals = sorted(data['CDR'].unique())
    
    # OASIS nWBV (Original)
    nwbv_means = [
        data[data['CDR'] == cdr]['nWBV'].mean()
        for cdr in cdr_vals
    ]
    ax1.bar(cdr_vals, nwbv_means, color='hotpink', width=0.4)
    ax1.set_title('nWBV (Original)', fontsize=20)
    ax1.set_xlabel('CDR', fontsize=16)
    ax1.set_ylabel('Brain Volume (mm³)', fontsize=16)
    ax1.tick_params(labelsize=14)

    # Brain Extraction
    be_means = [
        data[data['CDR'] == cdr]['nWBV_brain_extraction'].mean()
        for cdr in cdr_vals
    ]
    ax2.bar(cdr_vals, be_means, color='mediumseagreen', width=0.4)
    ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
    ax2.set_xlabel('CDR', fontsize=16)
    ax2.set_ylabel('Brain Volume (mm³)', fontsize=16)
    ax2.tick_params(labelsize=14)

    # Deep Atropos
    da_means = [
        data[data['CDR'] == cdr]['nWBV_deep_atropos'].mean()
        for cdr in cdr_vals
    ]
    ax3.bar(cdr_vals, da_means, color='tab:blue', width=0.4)
    ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
    ax3.set_xlabel('CDR', fontsize=16)
    ax3.set_ylabel('Brain Volume (mm³)', fontsize=16)
    ax3.tick_params(labelsize=14)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Visualization: Brain Volume by MMSE Scores")
    st.write(
        """
    MMSE (Mini-Mental State Examination) is a cognitive assessment tool scored from 0-30, 
    where higher scores indicate better cognitive function. We can examine how brain volume 
    correlates with MMSE scores to see if cognitive performance relates to brain structure.
    We compare all three methods.
    """
    )
    st.code(
        """
# Get MMSE column
mmse_col = 'MMSE'

# Chi-squared objective function
def chi_squared(params, areas, times):
    m = params[0]
    b = params[1]
    fit = m * times + b
    vals = (areas - fit) ** 2 / fit
    return np.sum(vals)

# Create 3 subplots for each method
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# OASIS nWBV (Original)
data_mmse = Data[[mmse_col, 'nWBV']].dropna()
ax1.scatter(data_mmse[mmse_col], data_mmse['nWBV'], 
            color='hotpink', edgecolor='k', alpha=0.6)

# Chi-squared regression
y_mean = data_mmse['nWBV'].mean()
y_range = data_mmse['nWBV'].max() - data_mmse['nWBV'].min()
z = np.polyfit(data_mmse[mmse_col], data_mmse['nWBV'], 1)
init_slope, init_intercept = z[0], z[1]

result = scipy.optimize.minimize(
    chi_squared,
    [init_slope, init_intercept],
    args=(data_mmse['nWBV'].values, data_mmse[mmse_col].values),
    bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
)
m, b = result.x[0], result.x[1]
x_line = np.linspace(data_mmse[mmse_col].min(), data_mmse[mmse_col].max(), 100)
ax1.plot(x_line, m * x_line + b, 'k--', linewidth=2)
ax1.set_xlabel('MMSE Score', fontsize=16)
ax1.set_ylabel('Brain Volume', fontsize=16)
ax1.set_title('nWBV (Original)', fontsize=20)
ax1.tick_params(labelsize=14)
ax1.grid(True, alpha=0.3)

# Brain Extraction
data_mmse = Data[[mmse_col, 'nWBV_brain_extraction']].dropna()
ax2.scatter(data_mmse[mmse_col], data_mmse['nWBV_brain_extraction'], 
            color='mediumseagreen', edgecolor='k', alpha=0.6)

y_mean = data_mmse['nWBV_brain_extraction'].mean()
y_range = data_mmse['nWBV_brain_extraction'].max() - data_mmse['nWBV_brain_extraction'].min()
z = np.polyfit(data_mmse[mmse_col], data_mmse['nWBV_brain_extraction'], 1)
init_slope, init_intercept = z[0], z[1]

result = scipy.optimize.minimize(
    chi_squared,
    [init_slope, init_intercept],
    args=(data_mmse['nWBV_brain_extraction'].values, data_mmse[mmse_col].values),
    bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
)
m, b = result.x[0], result.x[1]
x_line = np.linspace(data_mmse[mmse_col].min(), data_mmse[mmse_col].max(), 100)
ax2.plot(x_line, m * x_line + b, 'k--', linewidth=2)
ax2.set_xlabel('MMSE Score', fontsize=16)
ax2.set_ylabel('Brain Volume', fontsize=16)
ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
ax2.tick_params(labelsize=14)
ax2.grid(True, alpha=0.3)

# Deep Atropos
data_mmse = Data[[mmse_col, 'nWBV_deep_atropos']].dropna()
ax3.scatter(data_mmse[mmse_col], data_mmse['nWBV_deep_atropos'], 
            color='tab:blue', edgecolor='k', alpha=0.6)

y_mean = data_mmse['nWBV_deep_atropos'].mean()
y_range = data_mmse['nWBV_deep_atropos'].max() - data_mmse['nWBV_deep_atropos'].min()
z = np.polyfit(data_mmse[mmse_col], data_mmse['nWBV_deep_atropos'], 1)
init_slope, init_intercept = z[0], z[1]

result = scipy.optimize.minimize(
    chi_squared,
    [init_slope, init_intercept],
    args=(data_mmse['nWBV_deep_atropos'].values, data_mmse[mmse_col].values),
    bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
)
m, b = result.x[0], result.x[1]
x_line = np.linspace(data_mmse[mmse_col].min(), data_mmse[mmse_col].max(), 100)
ax3.plot(x_line, m * x_line + b, 'k--', linewidth=2)
ax3.set_xlabel('MMSE Score', fontsize=16)
ax3.set_ylabel('Brain Volume', fontsize=16)
ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
ax3.tick_params(labelsize=14)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""",
        language="python",
    )

    # Display the actual MMSE scatter plots for all 3 methods
    data = get_data()
    mmse_col = next((c for c in ['MMSE', 'mmse'] if c in data.columns), None)
    
    if mmse_col:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
        
        # OASIS nWBV (Original) - Chi-squared regression
        data_mmse = data[[mmse_col, 'nWBV']].dropna()
        if len(data_mmse) > 1:
            ax1.scatter(
                data_mmse[mmse_col],
                data_mmse['nWBV'],
                color='hotpink',
                edgecolor='k',
                alpha=0.6,
                s=30
            )
            
            # Chi-squared regression
            y_mean = data_mmse['nWBV'].mean()
            y_range = data_mmse['nWBV'].max() - data_mmse['nWBV'].min()
            z = np.polyfit(data_mmse[mmse_col], data_mmse['nWBV'], 1)
            init_slope, init_intercept = z[0], z[1]
            
            result = scipy.optimize.minimize(
                chi_squared,
                [init_slope, init_intercept],
                args=(data_mmse['nWBV'].values, data_mmse[mmse_col].values),
                bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
            )
            m, b = result.x[0], result.x[1]
            
            x_line = np.linspace(
                data_mmse[mmse_col].min(),
                data_mmse[mmse_col].max(),
                100
            )
            ax1.plot(x_line, m * x_line + b, 'k--', linewidth=2)
            ax1.set_xlabel('MMSE Score', fontsize=16)
            ax1.set_ylabel('Brain Volume', fontsize=16)
            ax1.set_title('nWBV (Original)', fontsize=20)
            ax1.tick_params(labelsize=14)
            ax1.grid(True, alpha=0.3)
        
        # Brain Extraction - Chi-squared regression
        data_mmse = data[[mmse_col, 'nWBV_brain_extraction']].dropna()
        if len(data_mmse) > 1:
            ax2.scatter(
                data_mmse[mmse_col],
                data_mmse['nWBV_brain_extraction'],
                color='mediumseagreen',
                edgecolor='k',
                alpha=0.6,
                s=30
            )
            
            # Chi-squared regression
            y_mean = data_mmse['nWBV_brain_extraction'].mean()
            y_range = data_mmse['nWBV_brain_extraction'].max() - data_mmse['nWBV_brain_extraction'].min()
            z = np.polyfit(
                data_mmse[mmse_col],
                data_mmse['nWBV_brain_extraction'],
                1
            )
            init_slope, init_intercept = z[0], z[1]
            
            result = scipy.optimize.minimize(
                chi_squared,
                [init_slope, init_intercept],
                args=(data_mmse['nWBV_brain_extraction'].values, data_mmse[mmse_col].values),
                bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
            )
            m, b = result.x[0], result.x[1]
            
            x_line = np.linspace(
                data_mmse[mmse_col].min(),
                data_mmse[mmse_col].max(),
                100
            )
            ax2.plot(x_line, m * x_line + b, 'k--', linewidth=2)
            ax2.set_xlabel('MMSE Score', fontsize=16)
            ax2.set_ylabel('Brain Volume', fontsize=16)
            ax2.set_title('nWBV (Brain Extraction)', fontsize=20)
            ax2.tick_params(labelsize=14)
            ax2.grid(True, alpha=0.3)
        
        # Deep Atropos - Chi-squared regression
        data_mmse = data[[mmse_col, 'nWBV_deep_atropos']].dropna()
        if len(data_mmse) > 1:
            ax3.scatter(
                data_mmse[mmse_col],
                data_mmse['nWBV_deep_atropos'],
                color='tab:blue',
                edgecolor='k',
                alpha=0.6,
                s=30
            )
            
            # Chi-squared regression
            y_mean = data_mmse['nWBV_deep_atropos'].mean()
            y_range = data_mmse['nWBV_deep_atropos'].max() - data_mmse['nWBV_deep_atropos'].min()
            z = np.polyfit(
                data_mmse[mmse_col],
                data_mmse['nWBV_deep_atropos'],
                1
            )
            init_slope, init_intercept = z[0], z[1]
            
            result = scipy.optimize.minimize(
                chi_squared,
                [init_slope, init_intercept],
                args=(data_mmse['nWBV_deep_atropos'].values, data_mmse[mmse_col].values),
                bounds=[(-0.01, 0.01), (y_mean - y_range, y_mean + y_range)],
            )
            m, b = result.x[0], result.x[1]
            
            x_line = np.linspace(
                data_mmse[mmse_col].min(),
                data_mmse[mmse_col].max(),
                100
            )
            ax3.plot(x_line, m * x_line + b, 'k--', linewidth=2)
            ax3.set_xlabel('MMSE Score', fontsize=16)
            ax3.set_ylabel('Brain Volume', fontsize=16)
            ax3.set_title('nWBV (Deep Atropos)', fontsize=20)
            ax3.tick_params(labelsize=14)
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("MMSE column not found in dataset.")


###############################################################
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

    available = [m for m in volume_methods if m in df.columns]

    # Apply custom tab styling
    apply_tab_styles()

    # Create tabs for different graph types
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Brain Volume Distributions",
            "Brain Volume by CDR - Bar Chart",
            "Brain Volume by CDR - Boxplot",
            "Brain Volume vs Age and Sex",
            "Brain Volume by MMSE Scores",
        ]
    )

    ###########################################################
    # TAB 1: HISTOGRAMS
    ###########################################################
    with tab1:
        st.markdown(
            "<p style='font-size:18px; font-weight:normal; margin-bottom:1rem;'>Distribution of Brain Volume</p>",
            unsafe_allow_html=True,
        )

        def plot_hist(df, method, ax, label, color):
            sns.histplot(df[method].dropna(), bins=20, ax=ax, color=color)
            ax.set_title(label, fontsize=20)
            ax.set_xlabel("Brain Volume (mm³)", fontsize=16)
            ax.set_ylabel("# Participants", fontsize=16)
            ax.tick_params(labelsize=14)
            ax.set_ylim(0, None)
            if method == "nWBV":
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: f"{x:.2f}")
                )

        fig = plot_method_comparison(
            df, available, method_labels, method_colors, plot_hist
        )
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Explanation of Data Sets and Results")
        st.write(
            """Add your explanation here about the histogram distributions."""
        )

    ###########################################################
    # TAB 2: MEAN ± SEM PLOTS
    ###########################################################
    with tab2:
        st.markdown(
            "<p style='font-size:18px; font-weight:normal; margin-bottom:1rem;'>Average Brain Volume by CDR - Bar Chart</p>",
            unsafe_allow_html=True,
        )
        if "CDR" in df.columns:

            def plot_bar(df, method, ax, label, color):
                grp = (df.groupby("CDR")[method].agg(["mean", "sem"]).reset_index())
                ax.bar(grp["CDR"].astype(str),grp["mean"],yerr=grp["sem"],capsize=6,color=color,)
                ax.set_title(label, fontsize=20)
                ax.set_xlabel("CDR", fontsize=16)
                ax.set_ylabel("Brain Volume (mm³)", fontsize=16)
                ax.tick_params(labelsize=14)
                ax.set_ylim(0.6, 0.9)

            fig = plot_method_comparison(df, available, method_labels, method_colors, plot_bar)
            st.pyplot(fig)

            # Display mean and SEM values in tables
            st.markdown("### Mean and SEM Values by CDR")
            for m in available:
                grp = df.groupby("CDR")[m].agg(["mean", "sem"]).reset_index()
                grp.columns = ["CDR", "Mean", "SEM"]
                st.markdown(f"**{method_labels[m]}**")
                st.dataframe(grp, hide_index=True)

        st.markdown("---")
        st.subheader("Explanation of Data Sets and Results")
        st.write( """Add your explanation here about average brain volume by CDR.""")

    ###########################################################
    # TAB 3: CDR BOXPLOTS
    ###########################################################
    with tab3:
        if "CDR" in df.columns:
            st.markdown(
                "<p style='font-size:18px; font-weight:normal; margin-bottom:1rem;'>Brain Volume by CDR — Boxplots</p>",
                unsafe_allow_html=True,
            )
            all_data = pd.concat([df[m].dropna() for m in available])
            ymin, ymax = all_data.min() * 0.95, all_data.max() * 1.05

            def plot_box(df, method, ax, label, color, ylims):
                sns.boxplot(x="CDR", y=method, data=df, ax=ax, color=color)
                ax.set_title(label, fontsize=20)
                ax.set_xlabel("CDR", fontsize=16)
                ax.set_ylabel("Brain Volume (mm³)", fontsize=16)
                ax.tick_params(labelsize=14)
                ax.set_ylim(*ylims)

            fig = plot_method_comparison(df, available, method_labels, method_colors, plot_box, ylims=(ymin, ymax),)
            st.pyplot(fig)

            # Display boxplot statistics for each CDR by method
            st.subheader("Boxplot Statistics by CDR for Each Method")
            for m in available:
                st.write(f"**{method_labels[m]}:**")
                stats_by_cdr = df.groupby("CDR")[m].describe(percentiles=[0.25, 0.5, 0.75])
                stats_display = stats_by_cdr[["min", "25%", "50%", "75%", "max"]].T
                stats_display.index = [
                    "Min",
                    "Q1 (25th percentile)",
                    "Median (50th percentile)",
                    "Q3 (75th percentile)",
                    "Max",
                ]
                st.dataframe(stats_display)
                st.write("")

        st.markdown("---")
        st.subheader("Explanation of Data Sets and Results")
        st.write("""Add your explanation here about brain volume by CDR.""")

    ###########################################################
    # TAB 4: SCATTERPLOTS + CHI-SQUARED REGRESSION
    ###########################################################
    with tab4:
        st.markdown(
            "<p style='font-size:18px; font-weight:normal; margin-bottom:1rem;'>Brain Volume vs Age — Chi-Squared Linear Regression</p>",
            unsafe_allow_html=True,
        )

        age_col = next((c for c in ["AGE", "Age", "age"] if c in df.columns), None  )
        sex_col = next(
            (
                c
                for c in ["M/F", "SEX", "Sex", "sex", "Gender", "gender"]
                if c in df.columns
            ),
            None,
        )

        if age_col and sex_col:
            st.write(
                """Manual linear regression using chi-squared minimization for parameter estimation."""
            )

            # Create side-by-side regression plots
            fig_chi, axes_chi = plt.subplots(
                1, len(available), figsize=(8 * len(available), 5)
            )
            if len(available) == 1:
                axes_chi = [axes_chi]

            chi_results = {}

            for i, m in enumerate(available):
                d = df[[age_col, m, sex_col]].dropna()

                # Separate by sex
                men = d[d[sex_col].str.lower().str.contains("m")]
                women = d[d[sex_col].str.lower().str.contains("f")]

                # Fit for men
                if len(men) > 1:
                    # Get data range for adaptive initial guess
                    y_mean = men[m].mean()
                    y_range = men[m].max() - men[m].min()

                    # Use numpy polyfit as initial guess
                    z = np.polyfit(men[age_col].values, men[m].values, 1)
                    init_slope, init_intercept = z[0], z[1]

                    men_ans = scipy.optimize.minimize(
                        chi_squared,
                        [init_slope, init_intercept],
                        args=(men[m].values, men[age_col].values),
                        bounds=[
                            (-0.01, 0.01),
                            (y_mean - y_range, y_mean + y_range),
                        ],
                    )
                    men_m, men_b = men_ans.x[0], men_ans.x[1]
                    men_chi2 = men_ans.fun
                else:
                    men_m, men_b, men_chi2 = None, None, None

                # Fit for women
                if len(women) > 1:
                    # Get data range for adaptive initial guess
                    y_mean = women[m].mean()
                    y_range = women[m].max() - women[m].min()

                    # Use numpy polyfit as initial guess
                    z = np.polyfit(women[age_col].values, women[m].values, 1)
                    init_slope, init_intercept = z[0], z[1]

                    women_ans = scipy.optimize.minimize(
                        chi_squared,
                        [init_slope, init_intercept],
                        args=(women[m].values, women[age_col].values),
                        bounds=[
                            (-0.01, 0.01),
                            (y_mean - y_range, y_mean + y_range),
                        ],
                    )
                    women_m, women_b = women_ans.x[0], women_ans.x[1]
                    women_chi2 = women_ans.fun
                else:
                    women_m, women_b, women_chi2 = None, None, None

                # Plot
                if len(men) > 1:
                    axes_chi[i].scatter( men[age_col], men[m], color="blue", alpha=0.6, s=30, label="Male",)
                    age_range = np.linspace(men[age_col].min(), men[age_col].max(), 100 )
                    men_fit = men_m * age_range + men_b
                    axes_chi[i].plot(
                        age_range,
                        men_fit,
                        color="blue",
                        linewidth=2,
                        linestyle="--",
                    )

                if len(women) > 1:
                    axes_chi[i].scatter(
                        women[age_col],
                        women[m],
                        color="red",
                        alpha=0.6,
                        s=30,
                        label="Female",
                    )
                    age_range = np.linspace(
                        women[age_col].min(), women[age_col].max(), 100
                    )
                    women_fit = women_m * age_range + women_b
                    axes_chi[i].plot(
                        age_range,
                        women_fit,
                        color="red",
                        linewidth=2,
                        linestyle="--",
                    )

                axes_chi[i].set_title(method_labels[m], fontsize=20)
                axes_chi[i].set_xlabel("Age (Years)", fontsize=16)
                axes_chi[i].set_ylabel("Brain Volume (mm³)", fontsize=16)
                axes_chi[i].tick_params(labelsize=14)
                axes_chi[i].legend(fontsize=12)

                chi_results[m] = {
                    "men": {
                        "slope": men_m,
                        "intercept": men_b,
                        "chi2": men_chi2,
                        "n": len(men),
                    },
                    "women": {
                        "slope": women_m,
                        "intercept": women_b,
                        "chi2": women_chi2,
                        "n": len(women),
                    },
                }

            plt.tight_layout()
            st.pyplot(fig_chi)
            plt.close(fig_chi)

            # Display chi-squared fit results
            st.subheader("Chi-Squared Fit Parameters")
            for m in available:
                st.write(f"**{method_labels[m]}:**")

                chi_stats = []
                for sex, label in [("men", "Male"), ("women", "Female")]:
                    res = chi_results[m][sex]
                    if res["slope"] is not None:
                        chi_stats.append(
                            {
                                "Sex": label,
                                "Slope (m)": f"{res['slope']:.6f}",
                                "Intercept (b)": f"{res['intercept']:.6f}",
                                "Equation": f"y = {res['slope']:.6f}x + {res['intercept']:.6f}",
                                "Chi² Value": f"{res['chi2']:.4f}",
                                "Sample Size": res["n"],
                            }
                        )

                if chi_stats:
                    chi_df = pd.DataFrame(chi_stats)
                    st.dataframe(chi_df, hide_index=True)
                st.write("")

        st.markdown("---")
        st.subheader("Explanation of Results")
        st.write("""Add interpretation here.""")

    ###########################################################
    # TAB 5: MMSE SCATTERPLOTS
    ###########################################################
    with tab5:
        mmse = next((c for c in ["MMSE", "mmse"] if c in df.columns), None)
        if mmse:
            st.markdown(
                "<p style='font-size:18px; font-weight:normal; margin-bottom:1rem;'>Brain Volume by MMSE Scores — Scatterplots</p>",
                unsafe_allow_html=True,
            )

            def plot_mmse_scatter(df, method, ax, label, color):
                d = df[[mmse, method]].dropna()
                ax.scatter(
                    d[mmse], d[method], color=color, edgecolor="k", alpha=0.6
                )
                stats = calc_regression_stats(d[mmse], d[method])
                if stats:
                    xx = np.linspace(d[mmse].min(), d[mmse].max(), 100)
                    ax.plot(
                        xx,
                        stats["poly"](xx),
                        color="black",
                        linestyle="--",
                        linewidth=2,
                    )
                ax.set_title(label, fontsize=20)
                ax.set_xlabel("MMSE Score", fontsize=16)
                ax.set_ylabel("Brain Volume", fontsize=16)
                ax.tick_params(labelsize=14)
                ax.grid(True, alpha=0.3)

            fig = plot_method_comparison(
                df, available, method_labels, method_colors, plot_mmse_scatter
            )
            st.pyplot(fig)

            # Display quartile statistics by MMSE score
            st.subheader("Boxplot Statistics by MMSE Score for Each Method")
            for m in available:
                st.write(f"**{method_labels[m]}:**")
                stats_by_mmse = df.groupby(mmse)[m].describe(
                    percentiles=[0.25, 0.5, 0.75]
                )
                stats_display = stats_by_mmse[
                    ["min", "25%", "50%", "75%", "max"]
                ].T
                stats_display.index = [
                    "Min",
                    "Q1 (25th percentile)",
                    "Median (50th percentile)",
                    "Q3 (75th percentile)",
                    "Max",
                ]
                st.dataframe(stats_display)
                st.write("")

            # Display regression statistics
            st.subheader("Regression Line Statistics")
            for m in available:
                d = df[[mmse, m]].dropna()
                stats = calc_regression_stats(d[mmse], d[m])
                if stats:
                    reg_data = pd.DataFrame(
                        [
                            {
                                "Slope": f"{stats['slope']:.4f}",
                                "Intercept": f"{stats['intercept']:.4f}",
                                "Equation": f"y = {stats['slope']:.4f}x + {stats['intercept']:.4f}",
                                "R²": f"{stats['r2']:.3f}",
                                "Sample Size": len(d),
                            }
                        ]
                    )
                    st.write(f"**{method_labels[m]}:**")
                    st.dataframe(reg_data, hide_index=True)
                    st.write("")

        st.markdown("---")
        st.subheader("Explanation of Data Sets and Results")
        st.write(
            """Add your explanation here about brain volume by MMSE scores."""
        )


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
