import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import cleaning
import eda
import stats_tests as stt
from config import AI_LOG, DATASET_PAGE, ECHONEST_AUDIO, MODEL_RESULTS
from data_loader import (feature_columns, load_cleaning_log, load_echonest,
                         load_missing_summary, load_small_features, load_tracks)
from models import comparison_table, top_confusions

RESEARCH_QUESTION = ("Can pre-computed audio features alone predict a track's top-level genre, "
                     "and which genres are most often confused with each other?")

@st.cache_data(show_spinner="Loading tracks ...")
def tracks():
    return load_tracks()

@st.cache_data(show_spinner="Loading audio features ...")
def small():
    return load_small_features()

@st.cache_data
def echo():
    return load_echonest()

@st.cache_data
def missing():
    return load_missing_summary()

@st.cache_data
def clog():
    return load_cleaning_log()

@st.cache_data(show_spinner="Running statistical tests ...")
def test_energy(feature="energy"):
    return stt.energy_by_genre(echo(), feature)

@st.cache_data
def test_chi2():
    return stt.genre_era_chi2(tracks())

@st.cache_data
def test_spearman():
    return stt.audio_listens_spearman(echo(), ECHONEST_AUDIO)

@st.cache_data(show_spinner="Checking feature outliers ...")
def feature_outliers():
    s = small()
    X = s[feature_columns(s)]
    flags = cleaning.isolation_forest_flags(X)
    by_genre = pd.Series(flags, index=s.index).groupby(s["genre_top"]).mean().mul(100).round(2)
    return cleaning.feature_outlier_summary(X), int(flags.sum()), by_genre

@st.cache_resource(show_spinner=False)
def model_results():
    if MODEL_RESULTS.exists():
        return joblib.load(MODEL_RESULTS)
    return None

def show_fig(result):
    fig, caption, interp = result
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption(caption)
    st.markdown(f"**Interpretation.** {interp}")
    st.write("")

PRINT_MODE = {"on": False}

def table(df, hide_index=False, **kw):
    """Interactive table on normal pages; static full-width table when printing."""
    if not PRINT_MODE["on"]:
        st.dataframe(df, hide_index=hide_index, **kw)
        return
    sty = df if hasattr(df, "hide") and not isinstance(df, pd.DataFrame) else df.style
    if isinstance(df, pd.DataFrame):
        sty = sty.format(precision=3)
    st.table(sty.hide(axis="index") if hide_index else sty)

def pfmt(p):
    return stt.fmt_p(p)

def overview():
    st.header("1. Dataset overview")
    st.markdown(f"**Research question.** {RESEARCH_QUESTION}")
    st.markdown(
        "**Why it matters.** Streaming services need genre tags for millions of tracks. If audio features "
        "are associated strongly enough with genre, tags can be suggested automatically. Knowing which "
        "genres get confused shows where automatic tagging needs human review.")
    st.markdown(
        f"**Source.** FMA: A Dataset for Music Analysis (Defferrard et al., ISMIR 2017), "
        f"[{DATASET_PAGE}]({DATASET_PAGE}). Creative-Commons-licensed tracks from the Free Music Archive. "
        "Only the metadata archive (`fma_metadata.zip`) is used; no raw audio is needed because audio "
        "features are pre-computed.")
    t, s, e, log = tracks(), small(), echo(), clog()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracks (tracks.csv)", f"{log['raw_tracks_shape'][0]:,}")
    c2.metric("Raw columns", log["raw_tracks_shape"][1])
    c3.metric("Audio features", log["raw_features_shape"][1])
    c4.metric("Genres in fma_small", s["genre_top"].nunique())
    st.markdown("""
| File | Header rows | What it contains | Used for |
|---|---|---|---|
| `tracks.csv` | 2 (group, field) | Per-track metadata: genre, duration, listens, dates, license, train/val/test split, subset | Cleaning, EDA, chi-square test |
| `features.csv` | 3 (feature, statistic, coefficient) | 518 librosa audio features: 11 families (MFCC, chroma, spectral contrast, …) × 7 statistics | PCA, classification models |
| `echonest.csv` | 3 | Echonest/Spotify descriptors (energy, danceability, valence, …) for a subset of tracks | ANOVA, correlation |
| `genres.csv` | 1 | Genre hierarchy | Reference |
""")
    st.markdown(
        f"**Label variable.** `track.genre_top`: the single top-level genre. The classifier uses the "
        f"**fma_small** subset ({len(s):,} tracks, {s['genre_top'].nunique()} genres, "
        f"{s['genre_top'].value_counts().min():,}–{s['genre_top'].value_counts().max():,} per genre) "
        f"with the dataset's official training / validation / test split. "
        f"Echonest descriptors exist for {log['echonest_rows']:,} tracks, "
        f"{log['echonest_rows_with_genre']:,} of which have a top-level genre.")
    st.subheader("Sample rows (cleaned working table)")
    sample = t[["genre_top", "duration", "listens", "favorites", "release_year", "split", "subset"]].head(10)
    table(sample.astype({"duration": "Int64", "release_year": "Int64"}))
    st.subheader("Split sizes in fma_small")
    table(pd.crosstab(s["genre_top"], s["split"], margins=True))

def cleaning_section():
    st.header("2. Data cleaning & preprocessing")
    log, miss = clog(), missing()
    st.subheader("2.1 Reading the multi-row headers")
    st.markdown(
        "`tracks.csv` has two header rows (e.g. `track` / `genre_top`) and `features.csv` and `echonest.csv` "
        "have three (e.g. `mfcc` / `mean` / `01`). They are read with `pd.read_csv(header=[0, 1])` / "
        "`header=[0, 1, 2]` into pandas MultiIndex columns, then flattened to single names such as "
        "`mfcc_mean_01`. Date columns are parsed with `errors='coerce'`, so malformed dates become missing "
        "instead of crashing the load.")
    st.code("tracks   = pd.read_csv('tracks.csv',   index_col=0, header=[0, 1])\n"
            "features = pd.read_csv('features.csv', index_col=0, header=[0, 1, 2])\n"
            "features.columns = ['_'.join(c) for c in features.columns]", language="python")
    st.markdown(f"Duplicate track IDs found: **{log['duplicate_track_ids']}**.")

    st.subheader("2.2 Missing values")
    show_fig(eda.fig_missing(miss))
    st.markdown("**Handling decision for every column** (sorted by % missing):")
    table(miss[["column", "n_missing", "pct_missing", "decision"]], hide_index=True, height=320)
    st.markdown(f"""
Key decisions and why:
- **`genre_top` missing for {log['genre_top_missing']:,} tracks ({100 * log['genre_top_missing'] / log['tracks_rows']:.0f}%).**
  In FMA this means the track was tagged with several top-level genres, or none. The label is the target
  variable, so it cannot be imputed. These rows stay in the general EDA and are excluded only from
  genre-based tests and models.
- **Release year missing for {log['release_year_missing']:,} tracks.** Shown as *Unknown* and excluded only
  from year-based analyses. Imputing a year would invent a time trend. {log['release_year_implausible_set_missing']}
  implausible years (before 1900 or after 2017) were also set to missing.
- **Mostly-empty free-text columns** (composer, lyricist, publisher, …) are not used by any analysis, so they
  are not carried forward. Nothing needed is lost.
- **Audio features:** {log['features_missing_cells']} missing cells in `features.csv`, so no imputation is needed.
- **Empty-list tags** (`[]`) are counted as missing, since an empty tag list carries no information.
""")

    st.subheader("2.3 Outliers and anomalies")
    st.markdown(
        "Outliers are detected first and then handled case by case. They are removed only when the value "
        "is impossible, not merely unusual.")
    t = tracks()
    show_fig(eda.fig_duration(t, log))
    show_fig(eda.fig_listens(t))
    st.markdown("**Multivariate outliers in the 518 audio features.**")
    fo, n_iso, by_genre = feature_outliers()
    show_fig(eda.fig_feature_outliers(fo))
    st.markdown(
        f"An Isolation Forest (2% contamination) on the standardized features flags **{n_iso}** of "
        f"{len(small()):,} tracks as anomalous. The flag rate is highest for **{by_genre.idxmax()}** "
        f"({by_genre.max():.1f}%) and lowest for **{by_genre.idxmin()}** ({by_genre.min():.1f}%). Removing "
        "them would remove part of a genre's natural variation, so they are kept and the effect is noted.")
    table(by_genre.rename("% flagged by Isolation Forest").to_frame().T)

    st.subheader("2.4 Summary of preprocessing steps")
    table(pd.DataFrame([
        ("Flatten multi-row headers", "all files", "Single, readable column names"),
        ("Parse dates (coerce errors)", "tracks", "Enables release / upload year"),
        ("Duration = 0 → missing", f"{log['duration_invalid_set_missing']} tracks", "Impossible value"),
        ("Flag duration IQR outliers (log scale)", f"{log['duration_outliers_flagged']} tracks", "Kept; real long/short tracks"),
        ("log1p(listens)", "all tracks", "Fix extreme skew"),
        ("Flag listens IQR outliers (log scale)", f"{log['listens_outliers_flagged']} tracks", "Kept; real popular tracks"),
        ("Implausible release years → missing", f"{log['release_year_implausible_set_missing']} tracks", "Data-entry errors"),
        ("Release era bins", "tracks with year", "Needed for the chi-square test"),
        ("Standardize features (z-score)", "fma_small", "Fit on training split only; for LR, k-NN, PCA"),
    ], columns=["Step", "Rows affected", "Reason"]), hide_index=True)

def eda_section():
    st.header("3. Exploratory data analysis")
    t, s, e = tracks(), small(), echo()
    st.subheader("3.1 Descriptive statistics")
    table(eda.descriptive_stats(t))
    st.markdown(
        "Mean listens far above the median confirms the right skew seen in Section 2. Duration has a "
        "similar but milder skew.")

    st.subheader("3.2 Group-by: statistics per genre")
    gs = eda.genre_summary(t)
    table(gs)
    longest = gs["duration_median_s"].idxmax()
    shortest = gs["duration_median_s"].idxmin()
    popular = gs["listens_median"].idxmax()
    st.markdown(
        f"**Interpretation.** Tracks differ by genre in length and popularity. **{longest}** has the longest "
        f"median duration ({gs.loc[longest, 'duration_median_s']:.0f}s) and **{shortest}** the shortest "
        f"({gs.loc[shortest, 'duration_median_s']:.0f}s). **{popular}** has the highest median listens "
        f"({gs.loc[popular, 'listens_median']:,.0f}). Medians are reported alongside means because both "
        "variables are skewed.")

    st.markdown("**Mean Echonest audio descriptors per genre** (genres with ≥ 30 tracks):")
    eg = e.dropna(subset=["genre_top"]).assign(genre_top=lambda d: d["genre_top"].astype(str))
    keep = eg["genre_top"].value_counts()[lambda v: v >= 30].index
    table(eg[eg["genre_top"].isin(keep)].groupby("genre_top")[ECHONEST_AUDIO].mean().round(3))

    st.subheader("3.3 Class balance")
    show_fig(eda.fig_genre_counts(t))
    st.subheader("3.4 Over time")
    show_fig(eda.fig_years(t))
    show_fig(eda.fig_uploads_by_genre(t))
    st.subheader("3.5 Relationships between audio descriptors")
    show_fig(eda.fig_corr(e, ECHONEST_AUDIO))
    st.subheader("3.6 Do genres separate in feature space? (PCA)")
    show_fig(eda.fig_pca(s, feature_columns(s)))

def _test_block(res, extra=None):
    st.markdown(f"- **H0:** {res['h0']}\n- **H1:** {res['h1']}")
    st.markdown(f"**Why this test.** {res['why']}")
    if extra:
        extra()
    st.markdown(f"**Result and interpretation.** {res['interpretation']}")

def stats_section():
    st.header("4. Statistical analysis & hypothesis testing")
    st.markdown("Significance level α = 0.05. All conclusions describe **associations**, not causes.")

    feature = st.selectbox("Audio descriptor for Test 1", ECHONEST_AUDIO, index=ECHONEST_AUDIO.index("energy"),
                           key="t1_feature")
    r1 = test_energy(feature)
    st.subheader(f"Test 1: {r1['name']}")

    def t1_extra():
        st.markdown("**Assumption checks**")
        table(r1["assumptions"].style.format({"statistic": "{:.3f}", "p_value": lambda p: pfmt(p)}),
                     hide_index=True)
        st.markdown(f"**Test statistics** (primary test: *{r1['primary']}*)")
        table(r1["results"].style.format({"statistic": "{:.3f}", "p_value": lambda p: pfmt(p)}),
                     hide_index=True)
        show_fig(eda.fig_energy_box(r1, feature))
        with st.expander("Post-hoc comparisons (pairwise Mann-Whitney, Holm-corrected) and Tukey HSD"):
            table(r1["posthoc"].style.format({"median_diff": "{:+.3f}", "p_holm": lambda p: pfmt(p)}),
                         hide_index=True)
            table(r1["tukey"], hide_index=True)
    _test_block(r1, t1_extra)

    r2 = test_chi2()
    st.subheader(f"Test 2: {r2['name']}")

    def t2_extra():
        c1, c2, c3 = st.columns(3)
        c1.metric("chi²", f"{r2['chi2']:.1f}")
        c2.metric("p-value", pfmt(r2["p"]))
        c3.metric("Cramér's V", f"{r2['cramers_v']:.3f}")
        show_fig(eda.fig_chi2_residuals(r2))
    _test_block(r2, t2_extra)

    r3 = test_spearman()
    st.subheader(f"Test 3: {r3['name']}")

    def t3_extra():
        table(r3["results"].style.format({"rho": "{:+.3f}", "p_value": lambda p: pfmt(p),
                                                 "p_holm": lambda p: pfmt(p)}), hide_index=True)
        fig, cap, _ = eda.fig_spearman(r3)
        st.pyplot(fig, clear_figure=True)
        st.caption(cap)
    _test_block(r3, t3_extra)

def ml_section():
    st.header("5. Machine learning: predicting genre from audio features")
    res = model_results()
    if res is None:
        st.warning("No saved model results. Run `python src/train_models.py` (takes a few minutes), "
                   "then reload this page.")
        return
    st.markdown(f"""
**Setup.** {res['n_features']} pre-computed audio features → 8 top-level genres (fma_small).
Official split: {res['n_train']:,} training / {res['n_val']:,} validation / {res['n_test']:,} test tracks.
Hyperparameters are chosen by **validation macro-F1**, and each chosen model is scored **once** on the test split.

| Model | Role | Why |
|---|---|---|
| Logistic Regression | **Baseline** | Linear, interpretable, standard first model |
| k-NN | Baseline (reference) | Distance-based; from the original proposal |
| Random Forest | **Advanced** | Non-linear, handles correlated features, robust to outliers |
| Gradient Boosting (HistGradientBoosting) | **Advanced** | Boosted trees, usually the strongest model on tabular data |

**Metrics.** Accuracy; **macro-F1** (primary, weights every genre equally); balanced accuracy; top-2
accuracy (is the true genre among the two most likely?); per-genre precision/recall/F1; confusion matrix.
""")
    with st.expander("Validation results for every hyperparameter tried"):
        table(res["tuning"].round(3), hide_index=True)
    ctable = comparison_table(res)
    st.subheader("5.1 Test-set comparison")
    table(ctable.rename(columns={"chosen params": "params", "balanced_accuracy": "bal. acc",
                                       "top2_accuracy": "top-2 acc", "macro_f1": "macro-F1"}),
                 hide_index=True)
    show_fig(eda.fig_model_compare(ctable))

    models = list(res["models"])
    pick = st.selectbox("Model for the detailed view", models, index=models.index(res["best_model"]),
                        key="ml_model")
    st.subheader(f"5.2 Which genres get confused? ({pick})")
    show_fig(eda.fig_confusion(res, pick))
    st.markdown("**Most frequent confusions**")
    table(top_confusions(res, pick).style.format({"share of true genre": "{:.1%}"}), hide_index=True)
    show_fig(eda.fig_per_class_f1(res, pick))
    with st.expander("Full classification report"):
        table(res["models"][pick]["report"].round(3))
    st.subheader("5.3 Which feature families matter most?")
    show_fig(eda.fig_importance(res))

def ai_section():
    st.header("6. AI-assisted analysis")
    st.info("This section is filled in during step 4 of the plan: the LangChain + LLM API assistant, "
            "the prompts used for each major decision, and whether each AI suggestion was accepted, "
            "modified or rejected (with justification).")
    path = AI_LOG / "interactions.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        table(pd.DataFrame(rows), hide_index=True)

def conclusions():
    st.header("7. Conclusions")
    res = model_results()
    r1, r2, r3 = test_energy("energy"), test_chi2(), test_spearman()
    if res is not None:
        best = res["best_model"]
        bt = res["models"][best]["test"]
        base = res["models"]["Logistic Regression"]["test"]
        conf = top_confusions(res, best, 3)
        conf_txt = "; ".join(f"{r['true genre']} → {r['predicted as']} ({r['share of true genre']:.0%})"
                             for _, r in conf.iterrows())
        imp = ", ".join(res["importance"]["family"].head(3))
        cmp_txt = (", so the baseline was not beaten by the advanced models" if best == "Logistic Regression"
                   else f", compared with {base['macro_f1']:.3f} for the logistic-regression baseline")
        st.markdown(f"""
**RQ1: Can audio features alone predict genre?** Partly. The best model, **{best}**, reaches
**{bt['accuracy']:.1%} accuracy** and **macro-F1 {bt['macro_f1']:.3f}** on unseen tracks (chance = 12.5%)
{cmp_txt}. The true genre is in the
top two predictions {bt['top2_accuracy']:.0%} of the time. The features carry real genre information,
but not enough for fully automatic tagging.

**RQ2: Which genres get confused?** The most frequent confusions are {conf_txt}. Confusions concentrate in a
few pairs, which is where an automatic tagger would most need human review.

**Which features are associated with genre?** {imp} contribute most (grouped permutation importance).
""")
    st.markdown(f"""
**Statistical findings.**
- {r1['interpretation']}
- {r2['interpretation']}
- {r3['interpretation']}

**Limitations.**
- Every result is an **association**: these analyses cannot show that a feature *causes* a genre label.
- Genre labels come from artists and FMA editors, and genres such as Experimental or International are broad
  and loosely defined. Some "errors" may be label ambiguity rather than model failure.
- fma_small is balanced by design. Real catalogues are imbalanced (Figure 2), so precision on rare genres would be lower.
- Echonest descriptors exist for only some tracks, and that subset may not be representative.
- Free Music Archive content (Creative-Commons, mostly independent artists) may not generalise to commercial music.

**Future work.** Train on the larger fma_medium subset (16 genres); learn features directly from spectrograms
with a CNN; combine audio features with metadata (artist, tags).
""")