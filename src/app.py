"""FMA genre analysis: interactive Streamlit report.

Run from the repo root:   streamlit run src/app.py
"""
import streamlit as st

import report_sections as rs
from data_loader import processed_available

from config import GITHUB_REPO

st.set_page_config(page_title="FMA: Predicting Genre from Audio Features", layout="centered")

# Print layout for "Save as PDF": hide app chrome, use the full page width, wrap long
# code and tables, start each numbered section on a new page, and keep figures whole.
PRINT_CSS = """
<style>
@media print {
  @page { size: letter; margin: 0.6in; }
  [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"],
  [data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stElementToolbar"],
  [data-testid="stSelectbox"], [data-testid="stHeaderActionElements"],
  [data-testid="stElementContainer"]:has(hr) { display: none !important; }  /* section breaks replace dividers */
  [data-testid="stMain"], [data-testid="stAppViewContainer"] { margin: 0 !important; padding: 0 !important; }
  [data-testid="stMainBlockContainer"] { max-width: 100% !important; padding: 0 !important; }
  body { font-size: 11pt; }
  /* Streamlit stacks elements with flexbox; Chrome leaves blank pages when splitting flex
     containers, so lay them out as normal blocks for print */
  [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stVerticalBlock"] {
    display: block !important; height: auto !important; overflow: visible !important;
  }
  [data-testid="stElementContainer"] { margin-bottom: 0.6rem; }
  /* scrolling boxes (tables, code) can't be split across pages and leave blank pages */
  [data-testid="stMainBlockContainer"] * { overflow: visible !important; max-height: none !important; }
  /* inline boxes are unbreakable in Chrome; tables and code must be normal blocks to split */
  [data-testid="stTable"] table { display: table !important; }
  [data-testid="stCode"] code { display: block !important; }

  /* each numbered section (st.header = h2) after the first starts a new page */
  [data-testid="stElementContainer"]:has(h2) ~ [data-testid="stElementContainer"]:has(h2) {
    break-before: page;
  }
  /* keep headings with what follows, and figures / metrics / rows in one piece */
  [data-testid="stElementContainer"]:has(h2, h3) { break-after: avoid; }
  [data-testid="stImage"], [data-testid="stHorizontalBlock"], tr { break-inside: avoid; }
  [class*="st-key-fig-"] { break-inside: avoid; }  /* figure, caption and interpretation together */
  img { max-width: 100% !important; height: auto !important; }

  /* tables: smaller type and wrapping instead of being cut off at the page edge */
  [data-testid="stTable"] { overflow: visible !important; }
  [data-testid="stTable"] table { width: 100% !important; font-size: 8pt !important; }
  [data-testid="stTable"] th, [data-testid="stTable"] td { padding: 2px 4px !important; white-space: normal !important; }

  /* code and logged prompts: wrap long lines */
  [data-testid="stCode"] pre, [data-testid="stCode"] code, pre, code {
    white-space: pre-wrap !important; word-break: break-word !important; overflow: visible !important;
  }
  [data-testid="stCode"] pre { font-size: 7.5pt !important; }
}
</style>
"""
st.markdown(PRINT_CSS, unsafe_allow_html=True)

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
st.sidebar.caption("To export the PDF: open *Full report*, wait for everything to load, then press "
                   "Ctrl/Cmd+P → Save as PDF, paper size Letter, with *Background graphics* on.")

st.title("Can audio features predict a track's genre?")
st.caption("Free Music Archive (FMA) dataset · audio features → top-level genre")
st.markdown(f"**GitHub repository:** [{GITHUB_REPO}]({GITHUB_REPO})")

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
    st.divider()

st.markdown(f"**GitHub repository:** [{GITHUB_REPO}]({GITHUB_REPO})")