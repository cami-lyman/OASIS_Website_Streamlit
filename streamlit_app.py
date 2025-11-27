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
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
st.title('Examining the Relationship between Brain Volume and Dementia Diagnoses 😺')

# Slider for years
min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

# Make a stable list of available country codes
countries = list(gdp_df['Country Code'].unique())

if not countries:
    st.warning("No countries available in the dataset")

# Ensure the multiselect default values exist in the available countries
default_countries = [c for c in ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'] if c in countries]

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    default=default_countries)

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

cols = st.columns(4)

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_vals = first_year[first_year['Country Code'] == country]['GDP']
        last_vals = last_year[last_year['Country Code'] == country]['GDP']

        if first_vals.empty or last_vals.empty:
            first_gdp = float('nan')
            last_gdp = float('nan')
        else:
            first_gdp = first_vals.iat[0] / 1_000_000_000
            last_gdp = last_vals.iat[0] / 1_000_000_000

        if math.isnan(first_gdp) or math.isnan(last_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            if first_gdp == 0:
                growth = '∞'
            else:
                growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )

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

st.header("MRI Slice Viewer")

SLICE_DIR = "oasis/mri_files"
slice_files = sorted([f for f in os.listdir(SLICE_DIR) if f.lower().endswith(".png")])

# Initialize state for continuous slider
if "slice_index" not in st.session_state:
    st.session_state.slice_index = 0

# Continuous-update slider
idx = st.slider(
    "Slice Number",
    0,
    len(slice_files) - 1,
    value=st.session_state.slice_index,
    key="live_slider"
)

# Update the stored index as slider moves
st.session_state.slice_index = idx

# Display the current slice
img_path = os.path.join(SLICE_DIR, slice_files[st.session_state.slice_index])
image = Image.open(img_path)
st.image(image, caption=f"Slice {st.session_state.slice_index}", use_column_width=True)

# OPTIONAL — Add autoplay animation
st.subheader("Autoplay (Optional)")

if st.button("Play Through Slices"):
    for i in range(len(slice_files)):
        st.session_state.slice_index = i
        st.experimental_rerun()
