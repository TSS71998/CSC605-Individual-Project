import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---- palette (validated categorical order; one-hue sequential; blue<->red diverging)
CAT = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
BLUE, ORANGE, RED, GRAY = "#257de0", "#f47230", "#ec4647", "#b4b4b4"
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fe", "#cce3fd", "#84b9f3", "#338ced", "#1b64be", "#0e3d7e"])
DIV = LinearSegmentedColormap.from_list("div", ["#1757ac", "#84b9f3", "#efefef", "#f5a3a3", "#c42b2c"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "legend.frameon": False,
    "axes.axisbelow": True,
})

def _bar_style():
    return dict(edgecolor="white", linewidth=1.5)

def genre_summary(tracks):
    """Group-by table: per-genre counts and descriptive statistics."""
    d = tracks[tracks["genre_top"].notna()].copy()
    d["genre_top"] = d["genre_top"].astype(str)
    g = d.groupby("genre_top")
    out = pd.DataFrame({
        "tracks": g.size(),
        "artists": g["artist_id"].nunique(),
        "duration_mean_s": g["duration"].mean(),
        "duration_median_s": g["duration"].median(),
        "listens_median": g["listens"].median(),
        "listens_mean": g["listens"].mean(),
        "favorites_mean": g["favorites"].mean(),
        "release_year_median": g["release_year"].median(),
        "pct_duration_outliers": 100 * g["duration_outlier"].mean(),
    }).sort_values("tracks", ascending=False)
    return out.round(1)

def descriptive_stats(tracks):
    cols = ["duration", "listens", "log_listens", "favorites", "interest", "bit_rate", "release_year"]
    return tracks[cols].describe(percentiles=[.25, .5, .75, .99]).T.round(2)

def fig_missing(miss, top=20):
    d = miss.sort_values("pct_missing", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    used = ~d["decision"].str.startswith("Not used")
    ax.barh(d["column"], d["pct_missing"], color=np.where(used, BLUE, GRAY), height=0.7, **_bar_style())
    ax.set_xlabel("% of tracks missing")
    ax.set_xlim(0, 100)
    ax.set_title(f"Missing values: {top} most incomplete of {len(miss)} columns")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Used in analysis"), Patch(color=GRAY, label="Not used")],
              loc="lower right")
    fig.tight_layout()
    n_over50 = int((miss["pct_missing"] > 50).sum())
    used_cols = miss[~miss["decision"].str.startswith("Not used")]
    worst_used = used_cols.iloc[0] if len(used_cols) else None
    interp = (f"{n_over50} of {len(miss)} columns are more than half empty, mostly free-text metadata "
              "(composer, lyricist, publisher, tags) that the analysis doesn't need, so they are not carried "
              "forward. ")
    if worst_used is not None:
        interp += (f"Among the columns that are used, {worst_used['column']} has the most gaps "
                   f"({worst_used['pct_missing']:.1f}%). Each one is handled individually (see table), "
                   "rather than with a blanket dropna that would throw away most of the dataset.")
    return fig, "Figure 1. Share of missing values per column in tracks.csv (all tracks).", interp

def fig_genre_counts(tracks):
    vc = tracks["genre_top"].astype("string").value_counts()
    n_missing = int(tracks["genre_top"].isna().sum())
    fig, ax = plt.subplots(figsize=(8, 5.2))
    d = vc.iloc[::-1]
    ax.barh(d.index, d.values, color=BLUE, height=0.7, **_bar_style())
    for y, v in enumerate(d.values):
        ax.text(v, y, f" {v:,}", va="center", color=TEXT2, fontsize=8)
    ax.set_xlabel("Number of tracks")
    ax.set_title("Tracks per top-level genre (all tracks with a genre)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    ratio = vc.iloc[0] / vc.iloc[-1]
    interp = (f"The full dataset is highly imbalanced: {vc.index[0]} has {vc.iloc[0]:,} tracks, "
              f"{ratio:,.0f}× more than {vc.index[-1]} ({vc.iloc[-1]:,}). A further {n_missing:,} tracks "
              f"({100 * n_missing / len(tracks):.0f}%) have no single top-level genre. This imbalance is why the "
              "classifier uses fma_small, which has 1,000 tracks for each of 8 genres, so accuracy is not inflated "
              "by always predicting the majority genre.")
    return fig, "Figure 2. Class balance: track counts per top-level genre.", interp

def fig_duration(tracks, log):
    d = tracks["duration"].dropna()
    lo, hi = log["duration_iqr_bounds_sec"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.logspace(np.log10(max(d.min(), 1)), np.log10(d.max()), 60)
    ax.hist(d, bins=bins, color=BLUE, **_bar_style())
    ax.set_xscale("log")
    for x, lab in [(lo, f"lower IQR fence {lo:.0f}s"), (hi, f"upper IQR fence {hi / 60:.1f} min")]:
        ax.axvline(x, color=ORANGE, lw=2, ls="--")
        ax.text(x, ax.get_ylim()[1] * 0.95, f" {lab}", color=TEXT2, fontsize=8, va="top")
    ax.set_xlabel("Track duration (seconds, log scale)")
    ax.set_ylabel("Tracks")
    ax.set_title("Distribution of track duration")
    fig.tight_layout()
    interp = (f"Duration is right-skewed, so outliers are detected with the IQR rule on log(duration). "
              f"{log['duration_outliers_flagged']:,} tracks fall outside the fences "
              f"({lo:.0f}s to {hi / 60:.1f} min), and {log['duration_over_1h']:,} run longer than an hour "
              f"(likely DJ mixes or live sets). These are real recordings, so they are flagged rather than "
              f"deleted; tree models are insensitive to them. "
              f"{log['duration_invalid_set_missing']:,} tracks with a duration of 0 s cannot be valid audio, so "
              "they were set to missing.")
    return fig, "Figure 3. Track duration (log scale) with the IQR outlier fences.", interp

def fig_listens(tracks):
    d = tracks["listens"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].hist(d, bins=60, color=BLUE, **_bar_style())
    axes[0].set_title("Raw listens")
    axes[0].set_xlabel("Listens")
    axes[0].set_ylabel("Tracks")
    axes[1].hist(np.log1p(d), bins=60, color=BLUE, **_bar_style())
    axes[1].set_title("log(1 + listens)")
    axes[1].set_xlabel("log(1 + listens)")
    fig.tight_layout()
    sk_raw, sk_log = d.skew(), np.log1p(d).skew()
    interp = (f"Raw listens are extremely skewed (skewness {sk_raw:.1f}): a few tracks have huge play counts "
              f"(max {d.max():,.0f}, median {d.median():,.0f}). The log transform reduces skewness to "
              f"{sk_log:.2f} and gives a usable distribution. High-listen tracks are genuine popular songs, "
              "not errors, so they are kept and analysed on the log scale instead of being removed.")
    return fig, "Figure 4. Listens before and after a log transform.", interp

def fig_years(tracks):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ry = tracks["release_year"].dropna()
    axes[0].hist(ry, bins=np.arange(ry.min(), ry.max() + 2) - 0.5, color=BLUE, **_bar_style())
    axes[0].set_title("Album release year (known only)")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Tracks")
    uy = tracks["upload_year"].value_counts().sort_index()
    axes[1].bar(uy.index, uy.values, color=BLUE, width=0.75, **_bar_style())
    axes[1].set_title("Year uploaded to FMA")
    axes[1].set_xlabel("Year")
    fig.tight_layout()
    pct_unknown = 100 * tracks["release_year"].isna().mean()
    recent = (ry >= 2000).mean() * 100
    interp = (f"Release year is unknown for {pct_unknown:.0f}% of tracks; those are labelled 'Unknown' and "
              f"left out only of year-based analyses. Among known years, {recent:.0f}% of tracks were released "
              f"in 2000 or later, so FMA mostly covers recent independent music. Uploads peak in {uy.idxmax()}.")
    return fig, "Figure 5. Release year (left) and upload year (right).", interp

def fig_uploads_by_genre(tracks, top=5):
    d = tracks[tracks["genre_top"].notna()].copy()
    d["genre_top"] = d["genre_top"].astype(str)
    leaders = d["genre_top"].value_counts().head(top).index
    d["g"] = np.where(d["genre_top"].isin(leaders), d["genre_top"], "Other")
    pv = d.pivot_table(index="upload_year", columns="g", values="duration", aggfunc="size", fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, g in enumerate(list(leaders) + ["Other"]):
        if g not in pv:
            continue
        color = GRAY if g == "Other" else CAT[i]
        ax.plot(pv.index, pv[g], color=color, lw=2, marker="o", ms=4, label=g)
    ax.set_xlabel("Upload year")
    ax.set_ylabel("Tracks uploaded")
    ax.set_title(f"Tracks uploaded per year, top {top} genres")
    ax.legend(ncol=6, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.15))
    fig.tight_layout()
    growth = (pv.iloc[-1] - pv.iloc[0]).drop("Other", errors="ignore")
    interp = (f"Upload volume changes over time for every genre, but the ranking is fairly stable. "
              f"{growth.idxmax()} shows the largest change from {pv.index[0]} to {pv.index[-1]}. Because the "
              "genre mix shifts over time, the release-era chi-square test below checks whether genre and era "
              "are associated.")
    return fig, "Figure 6. Uploads over time by genre (top genres; the rest grouped as Other).", interp

def fig_corr(echo, cols):
    c = echo[cols + ["log_listens"]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(c, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c)), c.columns)
    ax.grid(False)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(c.iloc[i, j]) > 0.6 else TEXT)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    ax.set_title("Correlation between Echonest audio features")
    fig.tight_layout()
    m = c.where(~np.eye(len(c), dtype=bool)).abs().stack()
    (a, b) = m.idxmax()
    interp = (f"The strongest pair is {a} and {b} (rho = {c.loc[a, b]:.2f}). Strongly correlated features carry "
              "overlapping information, which matters for interpreting logistic-regression coefficients but "
              "not for tree models. The log_listens row previews the popularity correlations tested formally "
              "in the Statistical Analysis section.")
    return fig, "Figure 7. Spearman correlation matrix of the Echonest audio features (n = "f"{len(echo):,}).", interp

def pca_projection(small, feature_cols):
    X = StandardScaler().fit_transform(small[feature_cols])
    pca = PCA(n_components=50, random_state=0).fit(X)
    Z = pca.transform(X)[:, :2]
    return pca, Z

def fig_pca(small, feature_cols):
    pca, Z = pca_projection(small, feature_cols)
    cum = np.cumsum(pca.explained_variance_ratio_)
    genres = sorted(small["genre_top"].unique())
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 5)
    ax0 = fig.add_subplot(gs[:, 0:1])
    ax0.plot(np.arange(1, len(cum) + 1), cum * 100, color=BLUE, lw=2)
    ax0.set_xlabel("Components")
    ax0.set_ylabel("Cumulative % variance")
    ax0.set_title("Explained variance")
    ax0.set_ylim(0, 100)
    lim_x = np.percentile(Z[:, 0], [1, 99])
    lim_y = np.percentile(Z[:, 1], [1, 99])
    for k, g in enumerate(genres):
        ax = fig.add_subplot(gs[k // 4, 1 + k % 4])
        m = (small["genre_top"] == g).to_numpy()
        ax.scatter(Z[~m, 0], Z[~m, 1], s=2, color=GRAY, alpha=0.25, linewidths=0)
        ax.scatter(Z[m, 0], Z[m, 1], s=3, color=BLUE, alpha=0.6, linewidths=0)
        ax.set_title(g, fontsize=9)
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.tick_params(labelsize=7)
    fig.suptitle("PCA of the 518 standardized audio features (fma_small): each genre highlighted against all tracks",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=TEXT)
    fig.tight_layout()
    v2 = cum[1] * 100
    n90 = int(np.searchsorted(cum, 0.9) + 1) if cum[-1] >= 0.9 else None
    interp = (f"The first two components explain only {v2:.1f}% of the variance"
              + (f", and {n90} components are needed to reach 90%" if n90 else
                 f"; even 50 components reach only {cum[-1] * 100:.0f}%")
              + ". In the 2-D projection the genres overlap heavily, with no genre forming a separate cluster. "
                "So genre is not linearly separable in a few dimensions; non-linear models that use all "
                "518 features may do better.")
    return fig, "Figure 8. PCA explained variance (left) and PC1 vs PC2 small multiples by genre.", interp

def fig_feature_outliers(df):
    d = df.sort_values("pct_tracks_with_any")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(d["family"], d["pct_tracks_with_any"], color=BLUE, height=0.7, **_bar_style())
    ax.set_xlabel("% of tracks with at least one |z| > 4 value in this family")
    ax.set_title("Extreme feature values by feature family (fma_small)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    top = df.sort_values("pct_tracks_with_any").iloc[-1]
    interp = (f"{top['family']} has the most extreme values ({top['pct_tracks_with_any']:.1f}% of tracks have at "
              "least one). Kurtosis and skew statistics naturally produce long tails, so these are not "
              "measurement errors. Decision: keep them. The tree models are unaffected by scale, and the "
              "linear and k-NN models use standardized inputs.")
    return fig, "Figure 9. Share of tracks with extreme (|z| > 4) values per feature family.", interp