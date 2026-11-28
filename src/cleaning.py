import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import FEATURE_FAMILIES, RANDOM_STATE

LIST_COLS = {("track", "tags"), ("album", "tags"), ("artist", "tags"),
             ("track", "genres"), ("track", "genres_all")}

DECISIONS = {
    ("track", "genre_top"): "Kept. Missing = track has several top-level genres or none. "
                            "Rows are kept for EDA; excluded only from genre-based tests and "
                            "models (the label cannot be imputed).",
    ("track", "duration"): "Kept. Zero-length tracks are set to missing (invalid); very long "
                           "tracks are flagged, not removed.",
    ("track", "listens"): "Kept. Heavy right tail, so analysed on a log1p scale.",
    ("track", "favorites"): "Kept for group-by summaries.",
    ("track", "interest"): "Kept for group-by summaries.",
    ("track", "bit_rate"): "Kept (no missing values expected).",
    ("track", "license"): "Kept; missing filled with 'Unknown' (category, not imputed).",
    ("track", "language_code"): "Kept for reference; mostly missing so not modelled.",
    ("track", "date_created"): "Kept; upload date, used for upload-year trends.",
    ("track", "date_recorded"): "Kept for reference; mostly missing so not used.",
    ("album", "date_released"): "Kept; used for release year. Missing years shown as "
                                "'Unknown' and excluded only from year-based analyses.",
    ("album", "type"): "Kept for reference.",
    ("album", "listens"): "Kept for reference.",
    ("artist", "id"): "Kept (used to count artists per genre).",
    ("artist", "location"): "Kept for reference; free text, not modelled.",
    ("set", "split"): "Kept; official train/validation/test split.",
    ("set", "subset"): "Kept; selects the balanced 'small' subset.",
}
UNUSED = "Not used by the analysis (free text / IDs / mostly empty), so not carried forward."

def missing_summary(raw_tracks):
    """Missing count and % for every raw column, with the handling decision."""
    rows = []
    n = len(raw_tracks)
    for col in raw_tracks.columns:
        s = raw_tracks[col]
        missing = s.isna()
        if col in LIST_COLS:
            missing |= s.astype(str).str.strip().isin(["[]", ""])
        elif s.dtype == object:
            missing |= s.astype(str).str.strip().eq("")
        rows.append({
            "group": col[0], "field": col[1], "column": f"{col[0]}.{col[1]}",
            "dtype": str(s.dtype), "n_missing": int(missing.sum()),
            "pct_missing": round(100 * missing.sum() / n, 2),
            "decision": DECISIONS.get(col, UNUSED),
        })
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)

def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

def clean_tracks(tracks):
    """Apply the documented cleaning steps; return (clean_df, log_dict)."""
    t = tracks.copy()
    log = {"tracks_rows": int(len(t))}

    # --- label
    log["genre_top_missing"] = int(t["genre_top"].isna().sum())
    t["has_genre"] = t["genre_top"].notna()

    # --- duration: 0 s (or negative) is not a real track -> missing
    invalid = t["duration"] <= 0
    log["duration_invalid_set_missing"] = int(invalid.sum())
    t.loc[invalid, "duration"] = np.nan
    log_dur = np.log(t["duration"])
    lo, hi = iqr_bounds(log_dur.dropna())
    t["duration_outlier"] = (log_dur < lo) | (log_dur > hi)
    log["duration_iqr_bounds_sec"] = [round(float(np.exp(lo)), 1), round(float(np.exp(hi)), 1)]
    log["duration_outliers_flagged"] = int(t["duration_outlier"].sum())
    log["duration_over_1h"] = int((t["duration"] > 3600).sum())

    # --- listens: keep everything, analyse on log scale
    t["log_listens"] = np.log1p(t["listens"])
    lo, hi = iqr_bounds(t["log_listens"].dropna())
    t["listens_outlier"] = (t["log_listens"] < lo) | (t["log_listens"] > hi)
    log["listens_outliers_flagged"] = int(t["listens_outlier"].sum())
    log["listens_zero"] = int((t["listens"] == 0).sum())

    # --- years
    t["release_year"] = t["album_date_released"].dt.year  # dates parsed in load_raw_tracks
    implausible = (t["release_year"] < 1900) | (t["release_year"] > 2017)
    log["release_year_implausible_set_missing"] = int(implausible.sum())
    t.loc[implausible, "release_year"] = np.nan
    log["release_year_missing"] = int(t["release_year"].isna().sum())
    t["upload_year"] = t["date_created"].dt.year
    t["era"] = pd.cut(t["release_year"], bins=[1899, 1999, 2009, 2017],
                      labels=["Before 2000", "2000-2009", "2010-2017"])

    # --- categoricals
    t["license"] = t["license"].fillna("Unknown")
    t["genre_top"] = t["genre_top"].astype("category")
    t["subset"] = pd.Categorical(t["subset"], categories=["small", "medium", "large"],
                                 ordered=True)
    return t, log


def feature_family(col):
    for fam in sorted(FEATURE_FAMILIES, key=len, reverse=True):
        if col.startswith(fam + "_"):
            return fam
    return "other"


def feature_outlier_summary(X, z=4.0):
    """Share of values with |z| > threshold, per feature family."""
    Z = np.abs(StandardScaler().fit_transform(X))
    fam = pd.Series([feature_family(c) for c in X.columns])
    rows = []
    for f in fam.unique():
        cols = np.where(fam == f)[0]
        rows.append({"family": f, "n_features": len(cols),
                     "pct_values_|z|>4": round(100 * (Z[:, cols] > z).mean(), 3),
                     "pct_tracks_with_any": round(100 * (Z[:, cols] > z).any(axis=1).mean(), 2)})
    return pd.DataFrame(rows).sort_values("pct_values_|z|>4", ascending=False)


def isolation_forest_flags(X, contamination=0.02):
    """Multivariate anomaly detection on the standardized feature matrix."""
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=RANDOM_STATE)
    Xs = StandardScaler().fit_transform(X)
    return iso.fit_predict(Xs) == -1