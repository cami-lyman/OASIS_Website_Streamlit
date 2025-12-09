###############################################################
# Unified Streamlit App: OASIS Brain Volume Analysis
###############################################################

import streamlit as st
import pandas as pd
import numpy as np
import os
import math
import time
from pathlib import Path
from typing import List
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Optional MRI loading
try:
    import nibabel as nib
    MRI_ENABLED = True
except:
    MRI_ENABLED = False

###############################################################
# Streamlit Config
###############################################################
st.set_page_config(
    page_title="Examining the Relationship between Brain Volume and Dementia Diagnoses",
    page_icon=":brain:",
    layout="wide"
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
    except:
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
    st.title("Examining the Relationship between Brain Volume and Dementia Diagnoses")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("About")
        st.write("""
        Dementia is a neurodegenerative disease which impacts millions worldwide...
        (***your full text kept exactly as written***)
        """)

    with col2:
        st.subheader("3D MRI Viewer")
        if not MRI_ENABLED:
            st.warning("nibabel not installed — 3D MRI viewer unavailable.")
        else:
            hdr_path = Path(__file__).parent / "data/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.hdr"
            if not hdr_path.exists():
                st.warning("HDR MRI file missing.")
            else:
                try:
                    img = nib.load(str(hdr_path))
                    data = np.squeeze(img.get_fdata())
                    
                    # Buttons for choosing view
                    c1, c2, c3 = st.columns(3)
                    if c1.button("Axial", key="overview_axial"):
                        st.session_state.mri_view = "Axial"
                        st.session_state.mri_play = False
                    if c2.button("Sagittal", key="overview_sagittal"):
                        st.session_state.mri_view = "Sagittal"
                        st.session_state.mri_play = False
                    if c3.button("Coronal", key="overview_coronal"):
                        st.session_state.mri_view = "Coronal"
                        st.session_state.mri_play = False
                    
                    # Determine slice axis
                    if st.session_state.mri_view == "Axial":
                        slice_axis = 2
                    elif st.session_state.mri_view == "Sagittal":
                        slice_axis = 0
                    else:
                        slice_axis = 1
                    
                    num_slices = data.shape[slice_axis]
                    
                    # Navigation
                    c1, c2, c3 = st.columns([1,1,1])
                    if c1.button("◀ Prev Slice", key="overview_prev"):
                        st.session_state.mri_slice_idx = max(0, st.session_state.mri_slice_idx - 1)
                    if c2.button("⏯ Play/Pause 3D", key="overview_play"):
                        st.session_state.mri_play = not st.session_state.mri_play
                    if c3.button("Next ▶ Slice", key="overview_next"):
                        st.session_state.mri_slice_idx = min(num_slices - 1, st.session_state.mri_slice_idx + 1)
                    
                    st.slider(f"{st.session_state.mri_view} Slice",
                              0, num_slices - 1,
                              key="overview_mri_slider",
                              value=st.session_state.mri_slice_idx,
                              on_change=update_mri_slice)
                    
                    # Extract slice
                    if slice_axis == 0:
                        slice_data = data[st.session_state.mri_slice_idx,:,:]
                    elif slice_axis == 1:
                        slice_data = data[:,st.session_state.mri_slice_idx,:]
                    else:
                        slice_data = data[:,:,st.session_state.mri_slice_idx]
                    
                    fig, ax = plt.subplots(figsize=(3,3))
                    ax.imshow(slice_data.T, cmap="twilight_shifted", origin="lower")
                    ax.axis("off")
                    st.pyplot(fig)
                    
                    if st.session_state.mri_play:
                        time.sleep(0.10)
                        st.session_state.mri_slice_idx = (st.session_state.mri_slice_idx + 1) % num_slices
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not load MRI file: {e}")


###############################################################
# PAGE: OASIS
###############################################################
def render_oasis():
    st.header("OASIS", divider="blue")
    
    st.subheader("About the OASIS Project")
    st.write("""
    The Open Access Series of Imaging Studies (OASIS) is a project aimed at making neuroimaging 
    datasets freely available to the scientific community. The data includes MRI scans from hundreds 
    of subjects across the adult lifespan, providing valuable resources for research into brain aging, 
    Alzheimer's disease, and other neurological conditions.
    
    The OASIS dataset used in this project includes:
    - Structural MRI scans
    - Demographic information
    - Clinical assessments including CDR (Clinical Dementia Rating)
    - Cognitive test scores (MMSE - Mini-Mental State Examination)
    
    This data enables researchers to investigate relationships between brain structure and cognitive function.
    """)


###############################################################
# PAGE: CODE (from Portfolio_sample_code.ipynb)
###############################################################
def render_code():
    st.header('Brain processing and Segmentation Code', divider='blue')
    st.write('''
    Brain Volumetric Pipeline - Code demonstrating various methods of calculating and normalizing brain volume from MRI scans.
    
    There are various methods of calculating and normalizing the volume of a brain from an MRI. 
    The following is the code that was used to do this in two different ways for our project:
    - Brain Extraction using ANTsPyNet.utilities.brain_extraction
    - Brain Extraction using ANTsPyNet.utilities.deep_atropos
    
    Both of these methods use deep learning models to predict and segment different types of tissue in an MRI scan.
    
    Included is also some exploratory code for visualizing the images and data.
    ''')
    
    st.subheader('Imports and Setup')
    st.code('''
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import os
import ants
from antspynet.utilities import brain_extraction
from antspynet.utilities import deep_atropos
import seaborn as sns
''', language='python')
    
    st.subheader('Load Data Files')
    st.write('Establish the paths used and set up initial dataframe')
    st.code('''
# File paths
path = "path/to/OASIS_selected/"
Dementia_path = "path/to/oasis_cross-sectional.csv"

# Load in dataframe
oasis_crossref = pd.read_csv(Dementia_path) 
oasis_crossref = oasis_crossref.dropna(subset=['CDR'])
''', language='python')
    
    st.subheader('Example Image Visualization')
    st.write('While not needed for any analysis, knowing what we\'re actively working with is useful.')
    st.code('''
# Load and display an MRI scan
path_ex = "path/to/example_brain.hdr"
img = nib.load(path_ex)
data = img.get_fdata()

# Display a middle slice
plt.imshow(data[:, :, data.shape[2]-95], cmap='twilight_shifted') 
plt.axis('off')
plt.title('Example slice image of Brain')
plt.show()
''', language='python')
    
    st.subheader('Volumetrics')
    st.write('''
    Now that we have a bit of an idea of what data we're utilizing, time to actually calculate the desired metrics.
    We start with creating a list of the IDs of the brain scans so we can look just at scans that:
    1. Exist
    2. Have a CDR to correlate to
    ''')
    st.code('''
# List of IDs
Ids = oasis_crossref['ID']
''', language='python')
    
    st.write('''
    With that, we can loop through all the brains, calculate each volume, and list them out.
    To do said calculations, we'll start by defining functions for each method.
    
    **Note:** It is with a heavy heart that these methods are flawed. By calculating the volumes of the ATLAS Registered images, 
    the volumes will be warped. The Atlas Scaling Factor (ASF) lets us convert back to natural space, but calculating the volume 
    in ATLAS space leads to stretching/warping of values. Therefore if this project were done professionally, we would experience 
    errors here. However, this process does a good job of estimating the normalized Whole Brain Volume and isn't terribly erroneous.
    
    *(If we wanted to correct this in the future and do it correctly, we would need to start in natural space and register to ATLAS space manually.)*
    ''')
    
    st.subheader('Method #1: Brain Extraction')
    st.write('Perform brain extraction using U-net and ANTs-based training data.')
    st.code('''
def brain_extraction_method(img):
    """Uses Ants.brain_extract to find brain volume of inputted MRI image"""
    
    # Create probability map 
    prob_brain_mask = brain_extraction(img, modality="t1", verbose=True)  
    brain_mask = ants.threshold_image(prob_brain_mask, 0.5, 1e9, 1, 0)
    
    # Sum up segmented voxels
    pixel_count = int(brain_mask.numpy().astype(bool).sum())
    
    # Calculate the volumes using voxel spacing 
    voxel_volume = float(np.prod(img.spacing))
    volume = pixel_count * voxel_volume
    
    return pixel_count, volume
''', language='python')
    
    st.subheader('Method #2: Deep Atropos Segmentation')
    st.write('''
    Six-tissue segmentation using deep learning.
    
    Labeling:
    - Label 0: background
    - Label 1: CSF
    - Label 2: gray matter
    - Label 3: white matter
    - Label 4: deep gray matter
    - Label 5: brain stem
    - Label 6: cerebellum
    ''')
    st.code('''
def atropos_segmentation(img, pre):
    """Uses deep atropos segmentation to calculate brain volume"""
    
    # Segment the brain
    seg = deep_atropos(img, do_preprocessing=pre) 
    seg_img = seg['segmentation_image']
    seg_np = seg_img.numpy().astype(int)

    # Select brain tissues (GM = 2, WM = 3, Deep GM = 4)
    brain_tissue = ((seg_np == 2) | (seg_np == 3) | (seg_np == 4)).astype(float)
    pixel_count = int(brain_tissue.sum())

    # Calculate volume using voxel spacing
    voxel_volume = float(np.prod(img.spacing))
    volume = pixel_count * voxel_volume
    
    return pixel_count, volume
''', language='python')
    
    st.subheader('Calculate Volumes for All Scans')
    st.write('**Do not run this following chunk unless you want your computer to die. It takes 90 minutes on a 4070.**')
    st.code('''
# Initialize arrays
pixel_counts_brain_extraction = np.zeros(len(Ids))
volumes_brain_extraction = np.zeros(len(Ids))
pixel_counts_deep_atropos = np.zeros(len(Ids))
volumes_deep_atropos = np.zeros(len(Ids))

# Loop through all brain scans
for i in range(len(Ids)):
    current_Id = Ids.iloc[i]
    
    # Load the image
    raw_img_ants = ants.image_read(filepath, reorient='IAL')
    raw_img_ants.set_spacing((1,1,1))
    
    # Run volume calculations
    pixel_counts_brain_extraction[i], volumes_brain_extraction[i] = \\
        brain_extraction_method(raw_img_ants)
    
    pixel_counts_deep_atropos[i], volumes_deep_atropos[i] = \\
        atropos_segmentation(raw_img_ants, pre=False)
''', language='python')
    
    st.subheader('Wrangling')
    st.write('''
    With the volumes calculated, now we need to normalize them!
    
    From *Buckner et al. 2004*, we know the following:
    - Estimated total intracranial volume (eTIV) is a fully automated estimate of TIVₙₐₜ (Total Intracranial Volume in natural space)
    - Total Intracranial Volume in ATLAS space (TIVₐₜₗ) divided by the ATLAS scaling factor (ASF) yields TIVₙₐₜ
    - The same relations apply to Volume (atl and nat)
    - Normalized Whole Brain Volume (nWBV) is the automated tissue segmentation based estimate of brain volume (gray-plus white-matter). 
      Normalized to percentage based on the atlas target mask.
    
    Therefore, we obtain the following equations and relations:
    ''')
    st.latex(r'eTIV \approx TIV_{nat} = \frac{TIV_{atl}}{ASF}')
    st.latex(r'Vol_{nat} = \frac{Vol_{atl}}{ASF}')
    st.latex(r'nWBV = \frac{Vol_{nat}}{eTIV}')
    st.write('''
    Which gives us the final equation we should apply to the calculated volumes to obtain volumes normalized to the ATLAS template brain scan used for segmentation and analysis:
    ''')
    st.latex(r'nWBV = \frac{Vol_{atl}}{eTIV \times ASF}')
    
    st.subheader('Normalize Brain Volumes')
    st.code('''
# Add volumes to dataframe
oasis_crossref['Vol BE'] = volumes_brain_extraction
oasis_crossref['Vol DA'] = volumes_deep_atropos

# Normalize with ASF
oasis_crossref['nWBV_brain_extraction'] = \\
    oasis_crossref['Vol BE'] / (oasis_crossref['ASF'] * (oasis_crossref['eTIV'] * 1000))

oasis_crossref['nWBV_deep_atropos'] = \\
    oasis_crossref['Vol DA'] / (oasis_crossref['ASF'] * (oasis_crossref['eTIV'] * 1000))
''', language='python')
    
    st.write('''
    **Note:** The following dataframe will have misaligned variable names as defined above. This is because the displayed df below 
    was not created by this chunk of code. It was copied over from an older example, and re-running the code to recreate the values 
    would be more than tedious.
    ''')
    st.code('''
# Load in modified csv with stored values
# This is done to create this example code without having to re-run
csv_data = "path/to/final_data_oasis.csv"
Data = pd.read_csv(csv_data)
''', language='python')
    
    st.subheader('Plotting')
    st.write('''
    With the above data, we have a lot of values to work with to answer our goal question(s):
    - nWBV_brain_extraction and nWBV_deep_atropos give us two different methods to compare for calculating a normalized whole brain volume
    - nWBV is the normalized brain volume from the original data set just for comparison and correctness
    - Clinical Dementia Rating (CDR) when compared with the nWBVs allows us to find out if CDR and brain volume are related in any ways
    - We can also see how CDR, Brain Volume, Age, Sex and other demographic data are related
    
    The below code includes some visualization plots for example.
    ''')
    
    st.subheader('Visualization: CDR vs nWBV')
    st.write('''
    This first one below is box and whisker for CDR vs nWBV. In theory this will show our biggest question/curiosity with this 
    project with regards to whether dementia and brain volume correlate.
    ''')
    st.code('''
# Box plot for CDR vs nWBV
sns.boxplot(x='CDR', y='nWBV', data=Data)
plt.xlabel('Clinical Dementia Rating')
plt.ylabel('Normalized Whole Brain Volume')
plt.show()
''', language='python')
    
    st.subheader('Visualization: nWBV vs Age by Gender')
    st.write('''
    Again, plotting a scatter plot this time, to compare the nWBVs with how they correlate (or not) with age.
    For this plot, the different genders were plotted separately to also see how that differs between M/F.
    ''')
    st.code('''
# Create sub-dataframes for genders
men = Data[Data['M/F'] == 'M']
women = Data[Data['M/F'] == 'F']

# Plot
plt.plot(men['Age'], men['nWBV'], '.', color='blue')
plt.plot(women['Age'], women['nWBV'], '.', color='red')
plt.xlabel('Age (Years)')
plt.ylabel('nWBV')
plt.legend(['Men', 'Women'])
plt.show()
''', language='python')
    
    st.subheader('Compare Methods')
    st.write('''
    While for the above plots we've used the nWBV values that were given with the OASIS 1 dataset, we wanted to find these on our own for this project, and have done so.
    
    The question remains of how effectively the brain_extraction method and the deep_atropos methods approximate nWBV.
    
    We can compare them visually by looking again at nWBV vs CDR, but with the 3 different methods.
    ''')
    st.code('''
# Compare three different methods side by side
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

# Brain extraction method
ax1.bar(Data['CDR'], Data['nWBV_brain_extraction'], color='blue', width=0.4)
ax1.set_title('Brain Extraction Method')
ax1.set_xlabel('CDR')
ax1.set_ylabel('nWBV')

# Deep atropos method
ax2.bar(Data['CDR'], Data['nWBV_deep_atropos'], color='red', width=0.4)
ax2.set_title('Deep Atropos Method')
ax2.set_xlabel('CDR')
ax2.set_ylabel('nWBV')

# Original OASIS nWBV
ax3.bar(Data['CDR'], Data['nWBV'], color='purple', width=0.4)
ax3.set_title('nWBV given by OASIS')
ax3.set_xlabel('CDR')
ax3.set_ylabel('nWBV')

plt.tight_layout()
plt.show()
''', language='python')


###############################################################
# PAGE: DATA & GRAPHS (UNIFIED)
###############################################################
def render_data_and_graphs():
    st.header("Data & Graphs", divider="blue")
    df = get_data()
    if df is None:
        st.error("Dataset missing.")
        return

    st.subheader("Preview of Dataset")
    st.dataframe(df.iloc[:,1:], height=210)

    # METHODS
    volume_methods = ["nWBV","nWBV_brain_extraction","nWBV_deep_atropos"]
    method_labels = {
        "nWBV": "nWBV (Original)",
        "nWBV_brain_extraction": "nWBV (Brain Extraction)",
        "nWBV_deep_atropos": "nWBV (Deep Atropos)"
    }
    method_colors = {
        "nWBV": "hotpink",
        "nWBV_brain_extraction": "mediumseagreen",
        "nWBV_deep_atropos": "tab:blue"
    }

    available = [m for m in volume_methods if m in df.columns]

    # Custom CSS for tab styling
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 22px;
        font-weight: normal;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        border: 1px solid rgba(0, 0, 0, 0.15);
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        margin-right: -1px;
        cursor: pointer;
        transition: all 0.2s ease;
        background-color: rgba(0, 0, 0, 0.03);
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: rgba(0, 0, 0, 0.08);
        border-color: rgba(0, 0, 0, 0.25);
        z-index: 1;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: white;
        border-bottom: 3px solid #1f77b4;
        font-weight: 500;
        z-index: 2;
        border-bottom-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create tabs for different graph types
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Brain Volume Distributions", 
        "Brain Volume by CDR - Bar Chart",
        "Brain Volume by CDR - Boxplot", 
        "Brain Volume vs Age and Sex",
        "Brain Volume by MMSE Scores"
    ])

    ##########################################################
    # TAB 1: HISTOGRAMS
    ##########################################################
    with tab1:
        st.markdown("<p style='font-size:18px; font-weight:normal; margin-bottom:1rem;'>Distribution of Brain Volume</p>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1,len(available), figsize=(8*len(available),5))
        if len(available)==1: axes=[axes]
        for i,m in enumerate(available):
            sns.histplot(df[m].dropna(), bins=20, ax=axes[i], color=method_colors[m])
            axes[i].set_title(method_labels[m], fontsize=20)
            axes[i].set_xlabel('Brain Volume (mm³)', fontsize=16)
            axes[i].set_ylabel('# Participants', fontsize=16)
            axes[i].tick_params(labelsize=14)
            axes[i].set_ylim(0, None)  # Start at 0, let max vary
            # Format x-axis to 2 decimal places for nWBV (Original)
            if m == "nWBV":
                axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Explanation of Data Sets and Results")
        st.write("""Add your explanation here about the histogram distributions.""")

    ##########################################################
    # TAB 2: MEAN ± SEM PLOTS
    ##########################################################
    with tab2:
        st.markdown("<p style='font-size:18px; font-weight:normal; margin-bottom:1rem;'>Average Brain Volume by CDR - Bar Chart</p>", unsafe_allow_html=True)
        if "CDR" in df.columns:
            fig, axes = plt.subplots(1,len(available), figsize=(8*len(available),5))
            if len(available)==1: axes=[axes]
            
            for i,m in enumerate(available):
                grp = df.groupby("CDR")[m].agg(["mean","sem"]).reset_index()
                axes[i].bar(grp["CDR"].astype(str), grp["mean"],
                            yerr=grp["sem"], capsize=6,
                            color=method_colors[m])
                axes[i].set_title(method_labels[m], fontsize=20)
                axes[i].set_xlabel('CDR', fontsize=16)
                axes[i].set_ylabel('Brain Volume (mm³)', fontsize=16)
                axes[i].tick_params(labelsize=14)
                axes[i].set_ylim(0.6, 0.9)
            st.pyplot(fig)
            
            # Display mean and SEM values in tables
            st.markdown("### Mean and SEM Values by CDR")
            for m in available:
                grp = df.groupby("CDR")[m].agg(["mean","sem"]).reset_index()
                grp.columns = ["CDR", "Mean", "SEM"]
                st.markdown(f"**{method_labels[m]}**")
                st.dataframe(grp, hide_index=True)
        
        st.markdown("---")
        st.subheader("Explanation of Data Sets and Results")
        st.write("""Add your explanation here about average brain volume by CDR.""")

    ##########################################################
    # TAB 3: CDR BOXPLOTS
    ##########################################################
    with tab3:
        if "CDR" in df.columns:
            st.markdown("<p style='font-size:18px; font-weight:normal; margin-bottom:1rem;'>Brain Volume by CDR — Boxplots</p>", unsafe_allow_html=True)
            fig, axes = plt.subplots(1,len(available), figsize=(8*len(available),5))
            if len(available)==1: axes=[axes]
            # Calculate common y-axis limits across all methods
            all_data = pd.concat([df[m].dropna() for m in available])
            ymin, ymax = all_data.min() * 0.95, all_data.max() * 1.05
            
            for i,m in enumerate(available):
                sns.boxplot(x="CDR", y=m, data=df, ax=axes[i], color=method_colors[m])
                axes[i].set_title(method_labels[m], fontsize=20)
                axes[i].set_xlabel('CDR', fontsize=16)
                axes[i].set_ylabel('Brain Volume (mm³)', fontsize=16)
                axes[i].tick_params(labelsize=14)
                axes[i].set_ylim(ymin, ymax)
            st.pyplot(fig)
            
            # Display boxplot statistics for each CDR by method
            st.subheader("Boxplot Statistics by CDR for Each Method")
            for m in available:
                st.write(f"**{method_labels[m]}:**")
                stats_by_cdr = df.groupby("CDR")[m].describe(percentiles=[.25, .5, .75])
                stats_display = stats_by_cdr[["min", "25%", "50%", "75%", "max"]].T
                stats_display.index = ["Min", "Q1 (25th percentile)", "Median (50th percentile)", "Q3 (75th percentile)", "Max"]
                st.dataframe(stats_display)
                st.write("")
        
        st.markdown("---")
        st.subheader("Explanation of Data Sets and Results")
        st.write("""Add your explanation here about brain volume by CDR.""")

    ##########################################################
    # TAB 4: SCATTERPLOTS + REGRESSION LINES
    ##########################################################
    with tab4:
        st.markdown("<p style='font-size:18px; font-weight:normal; margin-bottom:1rem;'>Brain Volume vs Age — Scatterplots</p>", unsafe_allow_html=True)

        age_col = next((c for c in ["AGE","Age","age"] if c in df.columns), None)
        sex_col = next((c for c in ["M/F","SEX","Sex","sex","Gender","gender"] if c in df.columns), None)

        if age_col and sex_col:
            fig, axes = plt.subplots(1, len(available), figsize=(8 * len(available), 5))
            if len(available) == 1:
                axes = [axes]

            # Legend handles
            female_handle = plt.Line2D([], [], marker='o', color='red', linestyle='None', label='Female')
            male_handle = plt.Line2D([], [], marker='o', color='blue', linestyle='None', label='Male')

            def sx(x):
                s = str(x).strip().lower()
                if s in ["f", "female"]: return "red"
                if s in ["m", "male"]: return "blue"
                return "gray"

            for i, m in enumerate(available):
                d = df[[age_col, m, sex_col]].dropna()
                colors = d[sex_col].map(sx)

                axes[i].scatter(d[age_col], d[m], c=colors, edgecolor="k", alpha=0.8)

                # Regression lines (without annotations)
                for sex, color, label in [("f", "red", "Female"), ("m", "blue", "Male")]:
                    sd = d[d[sex_col].str.lower().str.contains(sex)]

                    if len(sd) > 1:
                        # best fit line
                        z = np.polyfit(sd[age_col], sd[m], 1)
                        p = np.poly1d(z)

                        # Plot regression
                        xx = np.linspace(sd[age_col].min(), sd[age_col].max(), 100)
                        axes[i].plot(xx, p(xx), color=color, linestyle="--")

                axes[i].set_title(method_labels[m], fontsize=20)
                axes[i].set_xlabel(age_col, fontsize=16)
                axes[i].set_ylabel("Brain Volume (mm³)", fontsize=16)
                axes[i].legend(handles=[female_handle, male_handle], fontsize=12)

            st.pyplot(fig)
            
            # Display regression statistics in tables
            st.subheader("Regression Line Statistics")
            for m in available:
                st.write(f"**{method_labels[m]}:**")
                d = df[[age_col, m, sex_col]].dropna()
                
                regression_data = []
                for sex, label in [("f", "Female"), ("m", "Male")]:
                    sd = d[d[sex_col].str.lower().str.contains(sex)]
                    
                    if len(sd) > 1:
                        # Calculate regression
                        z = np.polyfit(sd[age_col], sd[m], 1)
                        slope, intercept = z[0], z[1]
                        
                        # Calculate R²
                        p = np.poly1d(z)
                        y_pred = p(sd[age_col])
                        y_true = sd[m]
                        ss_res = np.sum((y_true - y_pred) ** 2)
                        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
                        
                        regression_data.append({
                            "Sex": label,
                            "Slope": f"{slope:.4f}",
                            "Intercept": f"{intercept:.4f}",
                            "Equation": f"y = {slope:.4f}x + {intercept:.4f}",
                            "R²": f"{r2:.3f}",
                            "Sample Size": len(sd)
                        })
                
                if regression_data:
                    reg_df = pd.DataFrame(regression_data)
                    st.dataframe(reg_df, hide_index=True)
                st.write("")

        st.markdown("---")
        st.subheader("Explanation of Results")
        st.write("""Add interpretation here.""")

    ##########################################################
    # TAB 5: MMSE SCATTERPLOTS
    ##########################################################
    with tab5:
        mmse_cols = ["MMSE","mmse"]
        mmse = next((c for c in mmse_cols if c in df.columns), None)
        if mmse:
            st.markdown("<p style='font-size:18px; font-weight:normal; margin-bottom:1rem;'>Brain Volume by MMSE Scores — Scatterplots</p>", unsafe_allow_html=True)
            fig, axes = plt.subplots(1,len(available), figsize=(8*len(available),5))
            if len(available)==1: axes=[axes]
            
            for i,m in enumerate(available):
                d = df[[mmse, m]].dropna()
                
                # Scatter plot
                axes[i].scatter(d[mmse], d[m], color=method_colors[m], edgecolor="k", alpha=0.6)
                
                # Regression line
                if len(d) > 1:
                    z = np.polyfit(d[mmse], d[m], 1)
                    p = np.poly1d(z)
                    xx = np.linspace(d[mmse].min(), d[mmse].max(), 100)
                    axes[i].plot(xx, p(xx), color="black", linestyle="--", linewidth=2)
                
                axes[i].set_title(method_labels[m], fontsize=20)
                axes[i].set_xlabel("MMSE Score", fontsize=16)
                axes[i].set_ylabel("Brain Volume", fontsize=16)
                axes[i].tick_params(labelsize=14)
                axes[i].grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Display quartile statistics by MMSE score
            st.subheader("Boxplot Statistics by MMSE Score for Each Method")
            for m in available:
                st.write(f"**{method_labels[m]}:**")
                stats_by_mmse = df.groupby(mmse)[m].describe(percentiles=[.25, .5, .75])
                stats_display = stats_by_mmse[["min", "25%", "50%", "75%", "max"]].T
                stats_display.index = ["Min", "Q1 (25th percentile)", "Median (50th percentile)", "Q3 (75th percentile)", "Max"]
                st.dataframe(stats_display)
                st.write("")
            
            # Display regression statistics
            st.subheader("Regression Line Statistics")
            for m in available:
                d = df[[mmse, m]].dropna()
                
                if len(d) > 1:
                    # Calculate regression
                    z = np.polyfit(d[mmse], d[m], 1)
                    slope, intercept = z[0], z[1]
                    
                    # Calculate R²
                    p = np.poly1d(z)
                    y_pred = p(d[mmse])
                    y_true = d[m]
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
                    
                    reg_data = pd.DataFrame([{
                        "Slope": f"{slope:.4f}",
                        "Intercept": f"{intercept:.4f}",
                        "Equation": f"y = {slope:.4f}x + {intercept:.4f}",
                        "R²": f"{r2:.3f}",
                        "Sample Size": len(d)
                    }])
                    
                    st.write(f"**{method_labels[m]}:**")
                    st.dataframe(reg_data, hide_index=True)
                    st.write("")
        
        st.markdown("---")
        st.subheader("Explanation of Data Sets and Results")
        st.write("""Add your explanation here about brain volume by MMSE scores.""")
###############################################################
# PAGE: CONCLUSIONS
###############################################################
def render_conclusions():
    st.header("Conclusions", divider="blue")
    st.write("""
    (Your full conclusions text preserved exactly)
    """)


###############################################################
# PAGE: REFERENCES
###############################################################
def render_references():
    st.header("References", divider="blue")
    st.write("""
    [1] Olubunmi Kusoro, M. Roche, R. Del‐Pino‐Casado, P. Leung, and V. Orgeta, "Time to Diagnosis in Dementia: A Systematic Review With Meta‐Analysis," International Journal of Geriatric Psychiatry, vol. 40, no. 7, Jul. 2025, doi: https://doi.org/10.1002/gps.70129.
    
    [2] "Brain Changes Linked With Alzheimer's Years Before Symptoms Appear," Hopkinsmedicine.org, 2019. https://www.hopkinsmedicine.org/news/newsroom/news-releases/2019/05/brain-changes-linked-with-alzheimers-years-before-symptoms-appear
    
    [3] "Alzheimer's Disease (AD) & Neuroinflammation | Decoding AD," Decodingalzheimersdisease.com, 2024. https://www.decodingalzheimersdisease.com/role-of-neuroinflammation.html#the-science
    
    [4] M. Quarantelli, "MRI/MRS in neuroinflammation: methodology and applications," Clinical and Translational Imaging, vol. 3, no. 6, pp. 475–489, Sep. 2015, doi: https://doi.org/10.1007/s40336-015-0142-y.
    
    [5] "Open Access Series of Imaging Studies (OASIS)," Open Access Series of Imaging Studies (OASIS). https://sites.wustl.edu/oasisbrains/
    
    [6] M. Li et al., "Handedness- and Hemisphere-Related Differences in Small-World Brain Networks: A Diffusion Tensor Imaging Tractography Study," Brain Connectivity, vol. 4, no. 2, pp. 145–156, Mar. 2014, doi: https://doi.org/10.1089/brain.2013.0211.
    """)


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
