import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Examining the Relationship between Brain Volume and Dementia Diagnoses',
    page_icon=':brain:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Simple CSV loader



import streamlit as st
import pandas as pd
import math
from pathlib import Path
import os
from PIL import Image
import time
from typing import List
import numpy as np

# Try to import plotting libraries (optional for boxplot)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Examining the Relationship between Brain Volume and Dementia Diagnoses',
    page_icon=':brain:',
    layout='wide',
)

# -----------------------------------------------------------------------------
# Simple CSV loader


@st.cache_data
def get_data(filename=None):
    """Load a CSV file and return a DataFrame.

    By default this reads `data/final_data_oasis.csv` from the
    repository. If the file is missing, the caller can use `st.file_uploader`
    to provide a file at runtime.
    """

    if filename is None:
        filename = Path(__file__).parent / "data/final_data_oasis.csv"

    try:
        df = pd.read_csv(filename)
    except Exception:
        return None

    return df

# -----------------------------------------------------------------------------
# Draw the actual page (remove global title and old static sections)

# MRI Viewer setup (used on Overview page only)
SLICE_DIR = "oasis/mri_files"


@st.cache_data
def list_slices(slice_dir: str) -> List[str]:
    """Return sorted list of PNG slice filenames (reversed)."""
    try:
        files = [f for f in os.listdir(slice_dir) if f.lower().endswith(".png")]
        return sorted(files)[::-1]
    except Exception:
        return []


@st.cache_data
def load_image(path: str):
    """Load and return a PIL Image from `path`. Cached to avoid repeated disk I/O."""
    try:
        return Image.open(path)
    except Exception:
        return None

# Initialize state
if "slice_index" not in st.session_state:
    st.session_state.slice_index = 0
# keep a simplified play flag for UX, but we will not run a blocking autoplay loop
if "play" not in st.session_state:
    st.session_state.play = False

# --- CALLBACK FOR SLIDER ---
def update_slice():
    st.session_state.slice_index = st.session_state.slice_slider

def update_mri_slice():
    st.session_state.mri_slice_idx = st.session_state.mri_slider
    st.session_state.mri_play = False


def render_overview():
    st.title('Examining the Relationship between Brain Volume and Dementia Diagnoses')
    st.write('An analysis of data provided by the OASIS project.')
    
    # Create two columns: text on left, MRI viewer on right
    text_col, image_col = st.columns([1, 1])
    
    with text_col:
        st.subheader("About")
        st.write("""
        This project examines the relationship between brain volume measurements 
        and dementia diagnoses using data from the OASIS (Open Access Series of 
        Imaging Studies) project.
        
        The MRI viewer on the right displays cross-sectional brain scans that 
        allow you to explore the anatomy captured in this dataset.
        
        Use the controls to navigate through the slices and see how brain 
        structure varies across different levels.
        """)
    
    with image_col:
        st.subheader("MRI Slice Viewer")
        # Lazy-load the available slices (cached)
        slice_files = list_slices(SLICE_DIR)
        if not slice_files:
            st.warning('No MRI slices found in `oasis/mri_files`.')
            return

        max_idx = len(slice_files) - 1
        
        # Controls at the top
        col1, col2, col3 = st.columns([1, 1, 1])
        if col1.button("◀ Prev"):
            st.session_state.slice_index = max(0, st.session_state.slice_index - 1)
            st.session_state.play = False
        
        play_label = "⏸ Pause" if st.session_state.play else "▶ Play"
        if col2.button(play_label):
            st.session_state.play = not st.session_state.play
        
        if col3.button("Next ▶"):
            st.session_state.slice_index = min(max_idx, st.session_state.slice_index + 1)
            st.session_state.play = False

        # Slider to pick slice
        st.slider(
            "Slice",
            0,
            max_idx,
            value=st.session_state.slice_index,
            key="slice_slider",
            on_change=update_slice,
        )
        
        # Display the image
        img_path = os.path.join(SLICE_DIR, slice_files[st.session_state.slice_index])
        img = load_image(img_path)
        if img is None:
            st.error('Unable to load image.')
        else:
            st.image(img, caption=f"Slice {st.session_state.slice_index}", use_container_width=True)
    
    # Auto-advance if playing
    if st.session_state.play:
        time.sleep(0.15)
        st.session_state.slice_index = (st.session_state.slice_index + 1) % (max_idx + 1)
        st.rerun()

def render_oasis():
    st.header('OASIS', divider='blue')
    st.write("""
    The Open Access Series of Imaging Studies (OASIS) is a project aimed at making neuroimaging 
    datasets freely available to the scientific community. The data includes MRI scans from hundreds 
    of subjects across the adult lifespan.
    """)
    
    st.subheader("3D MRI Scan Viewer")
    
    # Try to load the HDR/IMG file pair
    try:
        import nibabel as nib
        
        hdr_path = Path(__file__).parent / "data/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.hdr"
        
        if not hdr_path.exists():
            st.warning("MRI scan file not found.")
            return
            
        # Load the 3D MRI volume
        img = nib.load(str(hdr_path))
        data = img.get_fdata()
        
        # Remove singleton dimensions and ensure we have a 3D volume
        data = np.squeeze(data)
        
        # Handle different possible orientations
        if data.ndim == 2:
            # If 2D after squeeze, add a dimension
            data = data[:, :, np.newaxis]
        elif data.ndim != 3:
            st.error(f"Unexpected data dimensions: {data.shape}")
            return
        
        # Initialize view orientation if not exists
        if "mri_view" not in st.session_state:
            st.session_state.mri_view = "Axial"
        
        # View selection buttons
        col1, col2, col3 = st.columns(3)
        if col1.button("Axial", width="stretch", type="primary" if st.session_state.mri_view == "Axial" else "secondary"):
            st.session_state.mri_view = "Axial"
            st.session_state.mri_play = False
        if col2.button("Sagittal", width="stretch", type="primary" if st.session_state.mri_view == "Sagittal" else "secondary"):
            st.session_state.mri_view = "Sagittal"
            st.session_state.mri_play = False
        if col3.button("Coronal", width="stretch", type="primary" if st.session_state.mri_view == "Coronal" else "secondary"):
            st.session_state.mri_view = "Coronal"
            st.session_state.mri_play = False
        
        # Determine slice parameters based on view
        if st.session_state.mri_view == "Axial":
            num_slices = data.shape[2]
            slice_axis = 2
        elif st.session_state.mri_view == "Sagittal":
            num_slices = data.shape[0]
            slice_axis = 0
        else:  # Coronal
            num_slices = data.shape[1]
            slice_axis = 1
        
        # Initialize slice index if not exists
        if "mri_slice_idx" not in st.session_state:
            st.session_state.mri_slice_idx = num_slices // 2
        
        # Initialize play state for OASIS viewer
        if "mri_play" not in st.session_state:
            st.session_state.mri_play = False
        
        # Reset slice index if it's out of bounds for current view
        if st.session_state.mri_slice_idx >= num_slices:
            st.session_state.mri_slice_idx = num_slices // 2
        
        # Play controls
        col1, col2, col3 = st.columns([1, 1, 1])
        if col1.button("◀ Prev ", key="mri_prev"):
            st.session_state.mri_slice_idx = max(0, st.session_state.mri_slice_idx - 1)
            st.session_state.mri_play = False
        
        play_label = "⏸ Pause " if st.session_state.mri_play else "▶ Play "
        if col2.button(play_label, key="mri_play_btn"):
            st.session_state.mri_play = not st.session_state.mri_play
        
        if col3.button(" Next ▶", key="mri_next"):
            st.session_state.mri_slice_idx = min(num_slices - 1, st.session_state.mri_slice_idx + 1)
            st.session_state.mri_play = False
        
        # Slider to navigate through slices (continuous updates)
        st.slider(
            f"Select MRI Slice ({st.session_state.mri_view} View)",
            0,
            num_slices - 1,
            value=st.session_state.mri_slice_idx,
            key="mri_slider",
            on_change=update_mri_slice
        )
        
        # Use the session state value for display
        slice_idx = st.session_state.mri_slice_idx
        
        # Extract the slice based on the axis
        if slice_axis == 0:
            slice_data = data[slice_idx, :, :]
        elif slice_axis == 1:
            slice_data = data[:, slice_idx, :]
        else:  # slice_axis == 2
            slice_data = data[:, :, slice_idx]
        
        # Create matplotlib figure 
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(slice_data.T, cmap='twilight_shifted', origin='lower')
        ax.set_title(f'MRI Slice {slice_idx}', fontsize=10)
        ax.axis('off')
        st.pyplot(fig, width='content')
        
        # Show volume dimensions
        st.info(f"Volume dimensions: {data.shape[0]} × {data.shape[1]} × {data.shape[2]} voxels")
        
        # Auto-advance if playing
        if st.session_state.mri_play:
            time.sleep(0.15)
            st.session_state.mri_slice_idx = (st.session_state.mri_slice_idx + 1) % num_slices
            st.rerun()
        
    except ImportError:
        st.error("nibabel library is required to load MRI files. Install it with: `pip install nibabel`")
    except Exception as e:
        st.error(f"Unable to load MRI scan: {e}")

def render_code():
    st.header('Code for Show', divider='blue')
    st.write('Key code snippets demonstrating the analysis techniques used in this project.')
    
    st.subheader('Data Loading and Preprocessing')
    st.code('''
import pandas as pd
import numpy as np

# Load OASIS dataset
df = pd.read_csv('data/final_data_oasis.csv')

# Compare three brain volume normalization methods
volume_methods = ['nWBV', 'nWBV_brain_extraction', 'nWBV_deep_atropos']
''', language='python')
    
    st.subheader('Statistical Analysis')
    st.code('''
# Calculate correlation between brain volume and cognitive scores
from scipy.stats import pearsonr

correlation, p_value = pearsonr(df['nWBV'], df['MMSE'])
print(f"Correlation: {correlation:.3f}, p-value: {p_value:.4f}")

# Group analysis by Clinical Dementia Rating (CDR)
grouped = df.groupby('CDR')['nWBV'].agg(['mean', 'std', 'count'])
''', language='python')
    
    st.subheader('Data Visualization')
    st.code('''
import matplotlib.pyplot as plt
import seaborn as sns

# Create boxplot comparing brain volumes by dementia severity
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='CDR', y='nWBV', data=df, ax=ax)
ax.set_xlabel('Clinical Dementia Rating')
ax.set_ylabel('Normalized Brain Volume')
plt.tight_layout()
''', language='python')
    
    st.subheader('Linear Regression Analysis')
    st.code('''
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
''', language='python')
    
    st.subheader('3D MRI Data Processing')
    st.code('''
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
''', language='python')

def render_data_and_graphs():
    st.header('Data & Graphs', divider='blue')
    df_local = get_data()
    if df_local is None:
        st.warning("No dataset found. Add `data/final_data_oasis.csv` to the `data/` folder.")
        return
    st.subheader("Data preview")
    st.dataframe(df_local.iloc[:, 1:], height=210)

    # Define the three brain volume methods to compare
    volume_methods = ['nWBV', 'nWBV_brain_extraction', 'nWBV_deep_atropos']
    method_labels = {
        'nWBV': 'nWBV (Original)',
        'nWBV_brain_extraction': 'nWBV (Brain Extraction)',
        'nWBV_deep_atropos': 'nWBV (Deep Atropos)'
    }
    
    if not PLOTTING_AVAILABLE:
        st.warning('Matplotlib and seaborn are required for plotting. Install them with: `pip install matplotlib seaborn`')
        return

    # Check which methods are available in the dataset
    available_methods = [m for m in volume_methods if m in df_local.columns]
    if not available_methods:
        st.warning(f'Dataset does not contain any of the brain volume columns: {", ".join(volume_methods)}')
        return

    # Define consistent colors for each method
    method_colors = {
        'nWBV': 'tab:orange',
        'nWBV_brain_extraction': 'tab:green',
        'nWBV_deep_atropos': 'tab:blue'
    }

    # Histograms: distribution of Brain Volume (MOVED TO FIRST)
    st.subheader('Distribution of Brain Volume (histogram) — Comparing Methods')
    try:
        fig, axes = plt.subplots(1, len(available_methods), figsize=(9*len(available_methods), 6))
        if len(available_methods) == 1:
            axes = [axes]
        
        for idx, method in enumerate(available_methods):
            vals = df_local[method].dropna()
            sns.histplot(vals, bins=20, kde=False, color=method_colors[method], ax=axes[idx])
            axes[idx].set_xlabel('Brain Volume', fontsize=14)
            axes[idx].set_ylabel('Count', fontsize=14)
            axes[idx].set_title(method_labels[method], fontsize=16)
            axes[idx].tick_params(axis='both', labelsize=12)
        
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f'Unable to render histograms: {e}')

    # Boxplots: Brain Volume by CDR for each method
    st.subheader('Brain Volume by CDR (box-and-whisker) — Comparing Methods')
    if 'CDR' in df_local.columns:
        try:
            fig, axes = plt.subplots(1, len(available_methods), figsize=(9*len(available_methods), 6))
            if len(available_methods) == 1:
                axes = [axes]
            
            for idx, method in enumerate(available_methods):
                sns.boxplot(x='CDR', y=method, data=df_local, ax=axes[idx], color=method_colors[method])
                axes[idx].set_xlabel('CDR', fontsize=14)
                axes[idx].set_ylabel('Brain Volume', fontsize=14)
                axes[idx].set_title(method_labels[method], fontsize=16)
                axes[idx].tick_params(axis='both', labelsize=12)
                # Zoom y-axis to emphasize differences
                try:
                    vals = df_local[method].dropna()
                    mean = vals.mean()
                    std = vals.std()
                    y_low = max(vals.min(), mean - 1.5 * std)
                    y_high = min(vals.max(), mean + 1.5 * std)
                    if y_high > y_low:
                        pad = (y_high - y_low) * 0.06
                        axes[idx].set_ylim(y_low - pad, y_high + pad)
                except Exception:
                    pass
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f'Unable to render boxplots: {e}')

        # Average Brain Volume by CDR (bar plots with standard error)
        try:
            st.subheader('Average Brain Volume by CDR (mean ± SEM) — Comparing Methods')
            fig, axes = plt.subplots(1, len(available_methods), figsize=(9*len(available_methods), 6))
            if len(available_methods) == 1:
                axes = [axes]
            
            for idx, method in enumerate(available_methods):
                grp = df_local.groupby('CDR')[method].agg(['mean', 'sem']).reset_index()
                axes[idx].bar(grp['CDR'].astype(str), grp['mean'], yerr=grp['sem'], 
                            capsize=6, color=method_colors[method])
                axes[idx].set_xlabel('CDR', fontsize=14)
                axes[idx].set_ylabel('Average Brain Volume', fontsize=14)
                axes[idx].set_title(method_labels[method], fontsize=16)
                axes[idx].tick_params(axis='both', labelsize=12)
                # Apply same zooming
                try:
                    vals = df_local[method].dropna()
                    mean = vals.mean()
                    std = vals.std()
                    y_low = max(vals.min(), mean - 1.5 * std)
                    y_high = min(vals.max(), mean + 1.5 * std)
                    if y_high > y_low:
                        pad = (y_high - y_low) * 0.06
                        axes[idx].set_ylim(y_low - pad, y_high + pad)
                except Exception:
                    pass
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f'Unable to render average brain volume plots: {e}')
    else:
        st.warning('Dataset does not contain `CDR` column.')

    # Boxplots: Brain Volume by MMSE
    st.subheader('Brain Volume by MMSE (Mini-Mental State Examination) — Comparing Methods')
    mmse_cols = ['MMSE', 'mmse', 'MiniMentalStateExamination']
    mmse_col = next((c for c in mmse_cols if c in df_local.columns), None)
    
    if mmse_col is not None:
        try:
            fig, axes = plt.subplots(1, len(available_methods), figsize=(9*len(available_methods), 6))
            if len(available_methods) == 1:
                axes = [axes]
            
            for idx, method in enumerate(available_methods):
                df_plot = df_local[[mmse_col, method]].dropna()
                
                # Create MMSE score groups for better visualization
                # MMSE ranges: 0-23 (severe), 24-27 (mild), 28-30 (normal)
                sns.boxplot(x=mmse_col, y=method, data=df_plot, ax=axes[idx], color=method_colors[method])
                
                axes[idx].set_xlabel('MMSE Score', fontsize=14)
                axes[idx].set_ylabel('Brain Volume', fontsize=14)
                axes[idx].set_title(method_labels[method], fontsize=16)
                axes[idx].tick_params(axis='both', labelsize=12)
                # Rotate x-axis labels if there are many unique values
                if df_plot[mmse_col].nunique() > 10:
                    axes[idx].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f'Unable to render MMSE boxplots: {e}')
    else:
        st.info('MMSE column not found in dataset. Skipping MMSE analysis.')

    # Boxplots: Brain Volume by Education Level
    st.subheader('Brain Volume by Education Level (years) — Comparing Methods')
    educ_cols = ['EDUC', 'Educ', 'Education', 'education', 'educ', 'YearsEducation']
    educ_col = next((c for c in educ_cols if c in df_local.columns), None)
    
    if educ_col is not None:
        try:
            fig, axes = plt.subplots(1, len(available_methods), figsize=(9*len(available_methods), 6))
            if len(available_methods) == 1:
                axes = [axes]
            
            for idx, method in enumerate(available_methods):
                df_plot = df_local[[educ_col, method]].dropna()
                
                # Use education years directly for boxplot grouping
                sns.boxplot(x=educ_col, y=method, data=df_plot, ax=axes[idx], color=method_colors[method])
                
                axes[idx].set_xlabel('Education Level (years)', fontsize=14)
                axes[idx].set_ylabel('Brain Volume', fontsize=14)
                axes[idx].set_title(method_labels[method], fontsize=16)
                axes[idx].tick_params(axis='both', labelsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f'Unable to render education level boxplots: {e}')
    else:
        st.info('Education column not found in dataset. Skipping education analysis.')

    # Scatterplots: Brain Volume vs Age, colored by sex
    st.subheader('Brain Volume vs Age (scatter) — Comparing Methods')
    age_cols = ['AGE', 'Age', 'age']
    sex_cols = ['M/F', 'SEX', 'Sex', 'sex', 'Gender', 'gender']
    age_col = next((c for c in age_cols if c in df_local.columns), None)
    sex_col = next((c for c in sex_cols if c in df_local.columns), None)

    if age_col is None or sex_col is None:
        st.warning('Dataset must contain age and sex columns to render the scatterplot.')
    else:
        try:
            fig, axes = plt.subplots(1, len(available_methods), figsize=(9*len(available_methods), 6))
            if len(available_methods) == 1:
                axes = [axes]
            
            def _sex_color(v):
                s = str(v).strip().lower()
                if s in ('f', 'female'):
                    return 'red'
                if s in ('m', 'male'):
                    return 'blue'
                return 'gray'

            for idx, method in enumerate(available_methods):
                df_plot = df_local[[age_col, method, sex_col]].dropna()
                colors = df_plot[sex_col].map(_sex_color)
                
                # Scatter plot for females and males separately for legend
                df_female = df_plot[df_plot[sex_col].str.strip().str.lower().isin(['f', 'female'])]
                df_male = df_plot[df_plot[sex_col].str.strip().str.lower().isin(['m', 'male'])]
                
                axes[idx].scatter(df_female[age_col], df_female[method], c='red', alpha=0.8, edgecolor='k', label='Female', s=50)
                axes[idx].scatter(df_male[age_col], df_male[method], c='blue', alpha=0.8, edgecolor='k', label='Male', s=50)
                
                # Add lines of best fit for males and females
                legend_lines = []
                for sex_val, color, label in [('f', 'red', 'Female'), ('m', 'blue', 'Male')]:
                    df_sex = df_plot[df_plot[sex_col].str.strip().str.lower().isin([sex_val, label.lower()])]
                    if len(df_sex) > 1:
                        # Calculate line of best fit
                        z = np.polyfit(df_sex[age_col], df_sex[method], 1)
                        p = np.poly1d(z)
                        
                        # Calculate R-squared
                        y_pred = p(df_sex[age_col])
                        y_actual = df_sex[method]
                        ss_res = np.sum((y_actual - y_pred) ** 2)
                        ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        x_line = np.linspace(df_sex[age_col].min(), df_sex[age_col].max(), 100)
                        line, = axes[idx].plot(x_line, p(x_line), color=color, linewidth=2, linestyle='--', alpha=0.7)
                        
                        # Create legend entry with equation
                        slope, intercept = z[0], z[1]
                        sign = '+' if intercept >= 0 else ''
                        legend_lines.append(f'{label} fit: y = {slope:.4f}x {sign}{intercept:.4f} (R² = {r_squared:.3f})')
                
                axes[idx].set_xlabel(age_col, fontsize=14)
                axes[idx].set_ylabel('Brain Volume', fontsize=14)
                axes[idx].set_title(method_labels[method], fontsize=16)
                axes[idx].tick_params(axis='both', labelsize=12)
                
                # Add legend with data points and fit equations
                handles, labels = axes[idx].get_legend_handles_labels()
                for line_text in legend_lines:
                    axes[idx].plot([], [], ' ', label=line_text)
                axes[idx].legend(loc='best', fontsize=10, framealpha=0.9)
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f'Unable to render scatterplots: {e}')

def render_conclusions():
    st.header('Conclusions', divider='blue')
    st.write('Summarize findings and takeaways here.')

def render_references():
    st.header('References', divider='blue')
    st.write('List references and citations here.')

# Sidebar navigation: custom link-based nav to allow exact styling (no circles)
st.sidebar.header('Navigation')

# Build pages list and read current page from query params (falls back to Overview)
PAGES = [
    'Overview',
    'OASIS',
    'Code',
    'Data & Graphs',
    'Conclusions',
    'References',
]

# Use radio for instant, in-process navigation (no query params used)
if 'page' not in st.session_state:
    st.session_state.page = 'Overview'

page = st.sidebar.radio('Navigation Menu', PAGES, index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0, label_visibility='collapsed')
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





