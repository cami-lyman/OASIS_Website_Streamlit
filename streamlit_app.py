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
# Draw the actual page

# Set the title that appears at the top of the page.
st.title('Examining the Relationship between Brain Volume and Dementia Diagnoses 😺')







# Additional project documentation sections
st.write("An analysis of data provided by the OASIS project.")
st.header('Our project', divider='blue')
st.write("Coroline's explanation of OASIS and what we are analyzing here.")
st.header('Background', divider='blue')
st.write("If needed and/or already written, background info will go here.")
st.header('Code', divider='blue')
st.write("Code nicely displayed below.")
st.header('Results', divider='blue')
st.header('Conclusions', divider='blue')
st.write("Final conclusions go here.")
st.header('References', divider='blue')
st.write("All references go here.")

st.title("MRI Slice Viewer")

# Show the CSV (uploaded or default)
uploaded = st.file_uploader("Upload a CSV to preview (optional)", type=["csv"]) 
data_source = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
    data_source = "(uploaded file)"
else:
    df = get_data()
    if df is not None:
        data_source = "data/Stuff_to_plot_and_play_with.csv"

if df is None:
    st.warning("No dataset found. Add `data/Stuff_to_plot_and_play_with.csv` or upload a CSV.")
else:
    st.info(f"Loaded dataset: {data_source}")
    st.subheader("Data preview (first 5 rows)")
    st.dataframe(df.head())

SLICE_DIR = "oasis/mri_files"

# --- REAL-TIME SLIDER ---
st.slider(
    "Slice",
    0,
    len(slice_files) - 1,
    value=st.session_state.slice_index,
    key="slice_slider",
    on_change=update_slice
)

# --- IMAGE PLACEHOLDER ---
image_placeholder = st.empty()

# Display current slice
img_path = os.path.join(SLICE_DIR, slice_files[st.session_state.slice_index])
img = Image.open(img_path)
image_placeholder.image(
    img,
    caption=f"Slice {st.session_state.slice_index}",
    width="stretch"
)

# --- PLAY/PAUSE BUTTONS ---
col1, col2 = st.columns(2)
if col1.button("▶ Play"):
    st.session_state.play = True
if col2.button("⏸ Pause"):
    st.session_state.play = False

# --- AUTOPLAY LOOP ---
if st.session_state.play:
    for i in range(st.session_state.slice_index, len(slice_files)):
        if not st.session_state.play:
            break

        st.session_state.slice_index = i

        # update image
        img_path = os.path.join(SLICE_DIR, slice_files[i])
        img = Image.open(img_path)
        image_placeholder.image(
            img,
            caption=f"Slice {i}",
            width="stretch"
        )

        # DO NOT WRITE TO st.session_state.slice_slider EVER
        # The slider will update on rerun because its value = slice_index

        time.sleep(0.12)








