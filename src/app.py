"""FMA genre analysis: interactive Streamlit report.

Run from the repo root:   streamlit run src/app.py
"""
import streamlit as st

import report_sections as rs
from data_loader import processed_available

st.set_page_config(page_title="FMA: Predicting Genre from Audio Features", layout="centered")

SECTIONS = {
    "1. Dataset overview": rs.overview,
    "2. Data cleaning": rs.cleaning_section,
    "3. Exploratory analysis": rs.eda_section,
    "4. Statistical tests": rs.stats_section,
    "5. Machine learning": rs.ml_section,
    "6. AI-assisted analysis": rs.ai_section,
    "7. Conclusions": rs.conclusions,
}

st.sidebar.title("FMA Genre Analysis")
st.sidebar.caption("CSC605 Individual Project")
choice = st.sidebar.radio("Section", list(SECTIONS) + ["Full report (all sections)"])
st.sidebar.markdown("---")
st.sidebar.caption("To export the PDF: open *Full report*, wait for everything to load, then use the "
                   "browser's Print → Save as PDF (or the ⋮ menu → Print).")

st.title("Can audio features predict a track's genre?")
st.caption("Free Music Archive (FMA) dataset · audio features to top-level genre")

if not processed_available():
    st.error("Processed data not found. From the repo root run:\n\n"
             "`python src/download_data.py`  then  `python src/train_models.py`")
    st.stop()

if choice.startswith("Full report"):
    rs.PRINT_MODE["on"] = True
    for fn in SECTIONS.values():
        fn()
        st.divider()
else:
    rs.PRINT_MODE["on"] = False
    SECTIONS[choice]()