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
# Simple CSV loader (not GDP-specific)



import streamlit as st
import pandas as pd
import math
from pathlib import Path
import os
from PIL import Image
import time
from typing import List

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
)

# -----------------------------------------------------------------------------
# Simple CSV loader (not GDP-specific)


@st.cache_data
def get_data(filename=None):
    """Load a CSV file and return a DataFrame.

    By default this reads `data/Stuff_to_plot_and_play_with.csv` from the
    repository. If the file is missing, the caller can use `st.file_uploader`
    to provide a file at runtime.
    """

    if filename is None:
        filename = Path(__file__).parent / "data/Stuff_to_plot_and_play_with.csv"

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


def render_overview():
    st.title('Examining the Relationship between Brain Volume and Dementia Diagnoses')
    st.write('An analysis of data provided by the OASIS project.')
    st.subheader("MRI Slice Viewer")
    # Lazy-load the available slices (cached)
    slice_files = list_slices(SLICE_DIR)
    if not slice_files:
        st.warning('No MRI slices found in `oasis/mri_files`.')
        return

    # Slider to pick slice (fast, but doesn't trigger blocking loops)
    max_idx = len(slice_files) - 1
    st.slider(
        "Slice",
        0,
        max_idx,
        value=st.session_state.slice_index,
        key="slice_slider",
        on_change=update_slice,
    )

    image_placeholder = st.empty()
    img_path = os.path.join(SLICE_DIR, slice_files[st.session_state.slice_index])
    img = load_image(img_path)
    if img is None:
        st.error('Unable to load image.')
    else:
        image_placeholder.image(img, caption=f"Slice {st.session_state.slice_index}", use_container_width=True)

    # Provide Next/Prev controls (non-blocking)
    col1, col2 = st.columns([1, 1])
    if col1.button("◀ Prev"):
        st.session_state.slice_index = max(0, st.session_state.slice_index - 1)
        st.experimental_rerun()
    if col2.button("Next ▶"):
        st.session_state.slice_index = min(max_idx, st.session_state.slice_index + 1)
        st.experimental_rerun()

def render_oasis():
    st.header('OASIS', divider='blue')
    st.write('Explain the OASIS project and dataset here.')

def render_code():
    st.header('Code', divider='blue')
    st.write('Below is the current app source:')
    try:
        src = Path(__file__).read_text()
        st.code(src, language='python')
    except Exception:
        st.warning('Unable to load source code preview.')

def render_data_and_graphs():
    st.header('Data & Graphs', divider='blue')
    df_local = get_data()
    if df_local is None:
        st.warning("No dataset found. Add `data/Stuff_to_plot_and_play_with.csv` to the `data/` folder.")
        return
    st.subheader("Data preview (first 5 rows)")
    st.dataframe(df_local.head())

    st.subheader('nWBV by CDR (box-and-whisker)')
    if not PLOTTING_AVAILABLE:
        st.warning('Matplotlib and seaborn are required for plotting. Install them with: `pip install matplotlib seaborn`')
        return
    if 'CDR' in df_local.columns and 'nWBV' in df_local.columns:
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x='CDR', y='nWBV', data=df_local, ax=ax)
            ax.set_xlabel('CDR')
            ax.set_ylabel('nWBV')
            ax.set_title('nWBV by CDR (box-and-whisker)')
            st.pyplot(fig)
        except Exception as e:
            st.error(f'Unable to render boxplot: {e}')
    else:
        st.warning('Dataset does not contain required columns: `CDR` and `nWBV`.')

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

# Sidebar CSS to make radio labels look like tiles and remove native radio visuals
st.sidebar.markdown(
    """
    <style>
    section[data-testid="stSidebar"] svg { display: none !important; }
    section[data-testid="stSidebar"] input[type="radio"] { display: none !important; }
    section[data-testid="stSidebar"] .stRadio label {
        display: block !important;
        padding: 10px 12px !important;
        margin: 4px 0 !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: background-color 0.12s ease !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(0,0,0,0.06) !important;
    }
    section[data-testid="stSidebar"] .stRadio [aria-checked="true"] {
        background: rgba(0,0,0,0.10) !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use radio for instant, in-process navigation (no query params used)
if 'page' not in st.session_state:
    st.session_state.page = 'Overview'

page = st.sidebar.radio('Go to', PAGES, index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0)
if page != st.session_state.page:
    st.session_state.page = page

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





