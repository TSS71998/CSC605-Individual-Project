# Predicting Music Genre from Audio Features (FMA)

CSC605 Individual Project: an end-to-end, reproducible data science workflow presented as an interactive Streamlit report.

## Dataset

**FMA: A Dataset for Music Analysis** (Defferrard, Benzi, Vandergheynst & Bresson, ISMIR 2017)
- Source: https://github.com/mdeff/fma
- File used: `fma_metadata.zip` (https://os.unil.cloud.switch.ch/fma/fma_metadata.zip, about 342 MB)
  - `tracks.csv`: 106,574 tracks × 52 metadata columns (2 header rows)
  - `features.csv`: 518 pre-computed librosa audio features per track (3 header rows)
  - `echonest.csv`: Echonest audio descriptors (energy, danceability, valence, …) for a subset of tracks
  - `genres.csv`: genre hierarchy
- No raw audio is needed.

## Project objective

**Research question:** Can pre-computed audio features alone predict a track's top-level genre, and which genres are most often confused with each other?

Music services need genre tags for millions of tracks. This project measures how strongly audio features are *associated* with genre, which genres are hard to separate, and which feature families matter most. The workflow covers cleaning (missing values and outliers), EDA with grouped statistics, hypothesis tests (ANOVA / Kruskal-Wallis, chi-square, Spearman), and classification with a baseline (Logistic Regression) and advanced models (Random Forest, Gradient Boosting).

## AI-assisted analysis

The Streamlit app integrates an LLM through **LangChain** (`ChatPromptTemplate | ChatGoogleGenerativeAI / ChatOpenAI | StrOutputParser`). For each major decision (header parsing, missing labels, outliers, skew, test choice, model and metric choice, interpretation), a prompt is filled with summary statistics computed from the data. The response is logged, and the author records whether the recommendation was **accepted, modified or rejected**, with a justification. Section 6 of the report also compares a naive prompt with a structured prompt.

- `ai_log/interactions.jsonl`: every prompt and response (timestamp, provider, model)
- `ai_log/decisions.json`: verdicts and justifications

**Important Note:** The logs in `ai_log/interactions.jsonl` are from an out dated version of the assistant that did not include timestamps, thus there are discrepancies between the logs and  the code format. They were kept as they are still pertient to the report as a whole.

## How to run

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Download the FMA metadata and build data/processed/  (one-time, a few minutes)
python src/download_data.py
#    If you already downloaded and unzipped the CSVs into data/raw/:
#    python src/download_data.py --skip-download

# 3. Train and evaluate the models (a few minutes; saves results/model_results.joblib)
python src/train_models.py

# 4. (Optional) export all figures as PNGs into figures/
python src/make_figures.py

# 5. Set up the AI assistant: copy .env.example to .env and add your API key
cp .env.example .env        # then edit .env (Gemini key: https://aistudio.google.com/apikey)

# 6. Launch the report
streamlit run src/app.py
```

The report runs without an API key. The saved AI interactions in `ai_log/` are still displayed; a key is only needed to ask the assistant new questions.

The processed data (`data/processed/`) and model results (`results/`) are committed, so step 6 works directly after cloning. Steps 2–4 reproduce them from scratch.

**PDF export:** in the app, choose **Full report (all sections)** in the sidebar, wait for it to load, then use the browser's *Print → Save as PDF*. The final PDF is in `report/`.

## Repository structure

```
data/
  raw/                  FMA CSVs (git-ignored; created by download_data.py)
  processed/            cleaned parquet files, missing-value summary, cleaning log
src/
  app.py                Streamlit entry point (navigation)
  report_sections.py    one render function per report section
  config.py             paths and constants
  download_data.py      download + extract + preprocess
  data_loader.py        read multi-row-header CSVs, flatten, write processed files
  cleaning.py           missing-value summary, outlier detection, cleaning decisions
  eda.py                figures (each with caption + data-driven interpretation) and group-by tables
  stats_tests.py        ANOVA / Welch / Kruskal-Wallis, chi-square, Spearman
  ai_assistant.py       LangChain LLM chain, decision prompts, interaction + verdict logging
  ai_context.py         builds the data-summary context sent with each prompt
  models.py             Logistic Regression, k-NN, Random Forest, Gradient Boosting
  train_models.py       trains all models -> results/model_results.joblib
  make_figures.py       exports every figure to figures/
figures/                exported PNG figures
results/                saved model metrics, confusion matrices, feature importance
report/                 final PDF report
ai_log/                 saved AI prompts, responses and verdicts
.env.example            template for the API key (copy to .env; .env is git-ignored)
```