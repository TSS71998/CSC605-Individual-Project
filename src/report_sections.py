"""One render function per report section. app.py calls them one at a time
(page navigation) or all in order (Full report, for the PDF export)."""
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import ai_assistant as ai
import ai_context
import cleaning
import eda
import stats_tests as stt
from config import DATASET_PAGE, ECHONEST_AUDIO, MODEL_RESULTS
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

def expander(label):
    """Collapsible on screen; always expanded when printing, so the content reaches the PDF."""
    if PRINT_MODE["on"]:
        st.markdown(f"**{label}**")
        return st.container()
    return st.expander(label)

def show_fig(result):
    fig, caption, interp = result
    with st.container(key="fig-" + caption.split(".")[0].split()[-1]):
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
        gh = r1["games_howell"]
        st.markdown(f"**Post-hoc (Games-Howell):** {int(gh['significant'].sum())} of {len(gh)} genre pairs differ "
                    "significantly. Full tables below.")
        with expander("Post-hoc comparisons: Games-Howell (primary) and Holm-corrected Mann-Whitney (robustness)"):
            table(gh.style.format({"mean_diff": "{:+.3f}", "se": "{:.4f}", "df": "{:.1f}",
                                   "p_value": lambda p: pfmt(p)}), hide_index=True)
            table(r1["posthoc"].style.format({"median_diff": "{:+.3f}", "p_holm": lambda p: pfmt(p)}),
                  hide_index=True)
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
        with st.container(key="fig-12"):
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
    with expander("Validation results for every hyperparameter tried"):
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
    with expander("Full classification report"):
        table(res["models"][pick]["report"].round(3))
    st.subheader("5.3 Which feature families matter most?")
    show_fig(eda.fig_importance(res))

def _load_secrets_into_env():
    """Allow keys in .streamlit/secrets.toml as well as .env."""
    try:
        for k in ["LLM_PROVIDER", "LLM_MODEL", "GOOGLE_API_KEY", "OPENAI_API_KEY"]:
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass

def _ai_data():
    return {"tracks": tracks(), "small": small(), "echo": echo(), "miss": missing(), "log": clog(),
            "energy": test_energy("energy"), "results": model_results()}

def _run(question, context, stage, decision_id, structured=True):
    try:
        with st.spinner("Asking the model ..."):
            ai.ask(question, context=context, stage=stage, decision_id=decision_id, structured=structured)
        st.rerun()
    except Exception as exc:  # show API / config errors instead of crashing the report
        st.error(f"LLM call failed: {exc}")

def _response_box(rec, empty_msg="Not run yet."):
    if rec is None:
        st.caption(empty_msg)
        return
    with st.container(border=True):
        st.markdown(rec["response"])
    st.caption(f"{rec.get('provider', '')} · {rec.get('model', '')}")

def _show_prompt(prompt):
    if PRINT_MODE["on"]:
        st.markdown("**Full prompt sent (system + user, with the data context):**")
        st.code(prompt, language="text")
    else:
        with st.expander("Full prompt sent (system message + data context)"):
            st.code(prompt, language="text")

def _decision_block(i, dec, data, verdicts):
    st.subheader(f"Decision {i}: {dec['title']}")
    st.caption(f"Stage: {dec['stage']}")
    st.markdown(f"**Prompt question.** {dec['question']}")
    rec = ai.latest(dec["id"])
    _show_prompt(rec["prompt"] if rec else ai.render_prompt(
        True, question=dec["question"], context=ai_context.build(dec["context"], data), stage=dec["stage"]))
    st.markdown("**AI response:**")
    _response_box(rec)
    st.markdown(f"**What the analysis implemented.** {dec['implemented']}")
    v = verdicts.get(dec["id"])
    if v:
        st.markdown(f"**Verdict: {v['verdict']}.** {v['justification']}")
    else:
        st.markdown("**Verdict:** not yet evaluated.")

def ai_section():
    st.header("6. AI-assisted analysis")
    _load_secrets_into_env()
    cfg = ai.settings()
    st.markdown(f"""
**How the assistant is integrated.** The app calls an LLM through **LangChain**
(`ChatPromptTemplate | chat model | StrOutputParser`) using the **{cfg['provider']}** API.
For each major decision, a prompt template is filled with **summary statistics computed from the data**
(never raw rows), and the model is asked for a recommendation in a fixed format: recommendation, reasoning,
assumptions/risks, and an alternative. Every prompt and response is saved to `ai_log/interactions.jsonl`.
After reading each response I compared it with the data and recorded whether I **accepted, modified or
rejected** it, and why (`ai_log/decisions.json`).
""")
    data = _ai_data()
    verdicts = ai.load_verdicts()
    log = ai.load_interactions()

    # ---- 6.1 prompt design
    st.subheader("6.1 Prompt design: naive vs. structured prompt")
    demo = next(d for d in ai.DECISIONS if d["id"] == ai.PROMPT_DEMO_ID)
    st.markdown(
        "The same question asked two ways. The **naive** prompt has no role, no data and no output format. "
        "The **structured** prompt gives the model a reviewer role, the actual data summary, the project goal "
        "and a required answer format, and tells it not to invent numbers.")
    naive_rec, struct_rec = ai.latest(demo["id"], "naive"), ai.latest(demo["id"], "structured")
    st.markdown(f"**Naive prompt:** `{demo['naive_question']}`")
    _response_box(naive_rec)
    st.markdown("**Structured prompt:** the same question with role, data context and format (full prompt in "
                "Decision 3 below).")
    _response_box(struct_rec)
    cmp = verdicts.get("prompt_comparison")
    if cmp:
        st.markdown(f"**Comparison.** {cmp['justification']}")

    # ---- 6.2 decisions
    st.subheader("6.2 AI assistance for each major decision")
    for i, dec in enumerate(ai.DECISIONS, 1):
        _decision_block(i, dec, data, verdicts)
        st.write("")

    # ---- 6.3 summary
    st.subheader("6.3 Summary of AI recommendations")
    summary = pd.DataFrame([{"decision": d["title"], "stage": d["stage"],
                             "verdict": verdicts.get(d["id"], {}).get("verdict", "not evaluated"),
                             "justification": verdicts.get(d["id"], {}).get("justification", "")}
                            for d in ai.DECISIONS])
    table(summary, hide_index=True)
    counts = summary["verdict"].value_counts()
    st.markdown("Verdict counts: " + ", ".join(f"**{k}** {v}" for k, v in counts.items()))

    # ---- 6.4 live assistant
    st.subheader("6.4 Ask the assistant")
    free = [r for r in log if r["decision_id"].startswith("free")]
    if not PRINT_MODE["on"]:
        if cfg["has_key"]:
            st.caption(f"Connected: {cfg['provider']} · {cfg['model']}")
        else:
            st.caption(f"No API key found. Add `{cfg['key_var'] or 'GOOGLE_API_KEY'}` to `.env` to ask questions.")
        sec = st.selectbox("Context to include", list(ai_context.SECTION_CONTEXT), key="free_section")
        q = st.text_area("Question (e.g. ask for code, a chart suggestion, or help interpreting a result)",
                         key="free_q")
        if st.button("Ask", key="ask_free", disabled=not cfg["has_key"] or not q.strip()):
            _run(q, ai_context.build_section(sec, data), sec, f"free:{sec}")
    elif not free:
        st.markdown("No additional questions were asked.")
    for r in free[::-1] if not PRINT_MODE["on"] else free:
        st.markdown(f"**Q ({r['stage']}):** {r['question']}")
        _response_box(r)

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