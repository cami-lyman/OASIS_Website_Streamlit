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
# Declare some useful functions.


@st.cache_data
   """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """
def get_brain_data():
    DATA_FILENAME = Path(__file__).parent/'data/Stuff_to_plot_and_play_with.csv'
    df = pd.read_csv(DATA_FILENAME)
    print(df.head())   # optional
    return df

   

gdp_df = get_gdp_data()

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

import streamlit as st
import os
from PIL import Image
import time

st.title("MRI Slice Viewer")

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








