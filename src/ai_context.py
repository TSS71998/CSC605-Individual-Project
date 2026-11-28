"""Builds the compact, numeric context passed to the LLM for each decision.
Only summaries are sent (counts, percentages, test statistics), never raw rows."""
from models import comparison_table, top_confusions

def _fmt_df(df, max_rows=15):
    return df.head(max_rows).to_string()

def headers(d):
    log = d["log"]
    return (f"tracks.csv: {log['raw_tracks_shape'][0]:,} rows x {log['raw_tracks_shape'][1]} columns, "
            "header rows = 2, e.g. ('track','genre_top'), ('album','listens'), ('set','split').\n"
            f"features.csv: {log['raw_features_shape'][1]} columns, header rows = 3, "
            "e.g. ('mfcc','mean','01'), ('chroma_cens','std','12').\n"
            "echonest.csv: header rows = 3, e.g. ('echonest','audio_features','energy').\n"
            "First column of every file is track_id; the row after the headers holds the index name.")

def missing(d):
    m = d["miss"][["column", "n_missing", "pct_missing"]]
    log = d["log"]
    return (f"Total tracks: {log['tracks_rows']:,}\n"
            f"genre_top (target label) missing: {log['genre_top_missing']:,} "
            f"({100 * log['genre_top_missing'] / log['tracks_rows']:.1f}%). In FMA this means the track has "
            "several top-level genres or none.\n"
            f"Release year missing: {log['release_year_missing']:,}\n"
            f"Missing cells in the 518 audio features: {log['features_missing_cells']}\n"
            "Missing % per column (top 15):\n" + _fmt_df(m, 15))

def duration(d):
    t, log = d["tracks"], d["log"]
    s = t["duration"].describe(percentiles=[.01, .25, .5, .75, .99]).round(1)
    return (f"Duration (seconds) after setting zero-length tracks to missing:\n{s.to_string()}\n"
            f"Zero-length tracks: {log['duration_invalid_set_missing']}\n"
            f"Tracks longer than 1 hour: {log['duration_over_1h']}\n"
            f"Skewness: {t['duration'].skew():.2f}\n"
            f"IQR fences computed on log(duration): {log['duration_iqr_bounds_sec']} seconds; "
            f"{log['duration_outliers_flagged']} tracks outside them.\n"
            "Duration is not an input to the genre classifier (which uses audio features only).")

def listens(d):
    t = d["tracks"]
    s = t["listens"].describe(percentiles=[.25, .5, .75, .99]).round(1)
    return (f"Listens per track:\n{s.to_string()}\n"
            f"Skewness raw: {t['listens'].skew():.1f}; skewness of log1p(listens): {t['log_listens'].skew():.2f}\n"
            f"Tracks with 0 listens: {int((t['listens'] == 0).sum())}\n"
            "Planned use: EDA and correlation with 8 Echonest audio features "
            f"(n = {d['echo']['log_listens'].notna().sum():,} tracks).")

def anova(d):
    r = d["energy"]
    a = r["assumptions"][["check", "statistic", "p_value"]].copy()
    res = r["results"][["test", "statistic", "df", "p_value", "effect_size"]]
    return ("Outcome: Echonest energy (0-1). Groups: top-level genres with >= 30 tracks.\n"
            "Group sizes and summary:\n" + _fmt_df(r["summary"].round(3), 30) + "\n\n"
            "Assumption checks:\n" + a.to_string(index=False) + "\n\n"
            "Test results:\n" + res.to_string(index=False))

def models(d):
    s = d["small"]
    counts = s["genre_top"].value_counts().to_string()
    split = s["split"].value_counts().to_string()
    return (f"Classification data: fma_small, {len(s):,} tracks, {s.shape[1] - 2} numeric audio features "
            "(11 families: MFCC, chroma x3, spectral contrast/centroid/bandwidth/rolloff, tonnetz, RMSE, ZCR; "
            "7 statistics each: mean, std, skew, kurtosis, min, max, median).\n"
            f"Tracks per genre:\n{counts}\n"
            f"Official split provided by the dataset:\n{split}\n"
            "Features are strongly correlated within each family. PCA: the first 2 components explain "
            "little of the variance and genres overlap in 2-D.")

def results(d):
    res = d["results"]
    if res is None:
        return "Model results not available yet."
    best = res["best_model"]
    rep = res["models"][best]["report"].loc[res["classes"], ["precision", "recall", "f1-score"]].round(2)
    imp = res["importance"][["family", "importance_mean"]].round(3)
    conf = top_confusions(res, best, 6)
    conf["share of true genre"] = (100 * conf["share of true genre"]).round(1)
    return ("Test-set comparison:\n" + comparison_table(res).to_string(index=False) + "\n\n"
            f"Best model: {best}. Per-genre metrics:\n{rep.to_string()}\n\n"
            "Most frequent confusions (% of true genre):\n" + conf.to_string(index=False) + "\n\n"
            "Grouped permutation importance (drop in macro-F1 when a feature family is shuffled):\n"
            + imp.to_string(index=False) + "\n\nChance accuracy for 8 balanced classes = 0.125.")

BUILDERS = {"headers": headers, "missing": missing, "duration": duration, "listens": listens,
            "anova": anova, "models": models, "results": results}

SECTION_CONTEXT = {  # used by the free-form question box
    "Data cleaning": ["missing", "duration"],
    "Exploratory analysis": ["listens", "duration"],
    "Statistical tests": ["anova"],
    "Machine learning": ["models", "results"],
}

def build(kind, data):
    return BUILDERS[kind](data)

def build_section(section, data):
    return "\n\n".join(build(k, data) for k in SECTION_CONTEXT[section])