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

# Reverse slice order
slice_files = sorted(
    [f for f in os.listdir(SLICE_DIR) if f.lower().endswith(".png")]
)[::-1]

# Initialize state
if "slice_index" not in st.session_state:
    st.session_state.slice_index = 0
if "play" not in st.session_state:
    st.session_state.play = False

# --- CALLBACK FOR SLIDER ---
def update_slice():
    st.session_state.slice_index = st.session_state.slice_slider


def render_overview():
    st.title('Examining the Relationship between Brain Volume and Dementia Diagnoses')
    st.write('An analysis of data provided by the OASIS project.')
    st.subheader("MRI Slice Viewer")
    # Re-render the MRI viewer controls and image
    st.slider(
        "Slice",
        0,
        len(slice_files) - 1,
        value=st.session_state.slice_index,
        key="slice_slider",
        on_change=update_slice
    )
    image_placeholder = st.empty()
    img_path = os.path.join(SLICE_DIR, slice_files[st.session_state.slice_index])
    img = Image.open(img_path)
    image_placeholder.image(
        img,
        caption=f"Slice {st.session_state.slice_index}",
        width="stretch"
    )
    col1, col2 = st.columns(2)
    if col1.button("▶ Play"):
        st.session_state.play = True
    if col2.button("⏸ Pause"):
        st.session_state.play = False
    if st.session_state.play:
        for i in range(st.session_state.slice_index, len(slice_files)):
            if not st.session_state.play:
                break
            st.session_state.slice_index = i
            img_path = os.path.join(SLICE_DIR, slice_files[i])
            img = Image.open(img_path)
            image_placeholder.image(img, caption=f"Slice {i}", width="stretch")
            time.sleep(0.12)

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

# Sidebar navigation
st.sidebar.header('Navigation')
# Sidebar styles matching streamlit_template - whole option darkens on hover/active
st.sidebar.markdown(
    """
    <style>
    /* Hide the radio button circles */
    section[data-testid="stSidebar"] .stRadio > div > label > div[data-testid="stMarkdownContainer"] {
        padding-left: 0;
    }
    section[data-testid="stSidebar"] .stRadio input[type="radio"] {
        display: none;
    }
    
    /* Style the label containers */
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 2px;
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 8px 12px;
        margin: 0;
        border-radius: 6px;
        background: transparent;
        transition: background-color 0.15s ease;
        cursor: pointer;
    }
    
    /* Hover state - slightly darker */
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(0, 0, 0, 0.06);
    }
    
    /* Active/selected state - darker background */
    section[data-testid="stSidebar"] .stRadio input:checked + div + label,
    section[data-testid="stSidebar"] .stRadio input:checked ~ label {
        background: rgba(0, 0, 0, 0.10);
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    'Go to',
    (
        'Overview',
        'OASIS',
        'Code',
        'Data & Graphs',
        'Conclusions',
        'References',
    )
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





