import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def p_eq(p):
    """'p < 0.001' or 'p = 0.042' for use inside sentences."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def welch_anova(groups):
    """Welch's one-way ANOVA (does not assume equal variances)."""
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups])
    w = n / v
    mw = np.sum(w * m) / np.sum(w)
    a = np.sum(w * (m - mw) ** 2) / (k - 1)
    tmp = np.sum((1 - w / np.sum(w)) ** 2 / (n - 1))
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
    F = a / b
    df1, df2 = k - 1, (k ** 2 - 1) / (3 * tmp)
    return F, df1, df2, stats.f.sf(F, df1, df2)

def games_howell(d, feature, order):
    """Games-Howell pairwise comparisons: no equal-variance or equal-n assumption
    (the standard post-hoc test after Welch's ANOVA)."""
    g = d.groupby("genre_top")[feature]
    n, m, v = g.size(), g.mean(), g.var(ddof=1)
    k = len(order)
    rows = []
    for a, b in itertools.combinations(order, 2):
        va, vb = v[a] / n[a], v[b] / n[b]
        se = np.sqrt(va + vb)
        diff = m[a] - m[b]
        df = (va + vb) ** 2 / (va ** 2 / (n[a] - 1) + vb ** 2 / (n[b] - 1))
        q = abs(diff) / se * np.sqrt(2)
        p = float(stats.studentized_range.sf(q, k, df))
        rows.append({"group1": a, "group2": b, "mean_diff": diff, "se": se, "df": df,
                     "p_value": p, "significant": p < ALPHA})
    return pd.DataFrame(rows)

def energy_by_genre(echo, feature="energy", min_n=30):
    """Test 1: is mean `feature` the same across top-level genres?"""
    d = echo.dropna(subset=["genre_top", feature]).copy()
    d["genre_top"] = d["genre_top"].astype(str)
    counts = d["genre_top"].value_counts()
    keep = counts[counts >= min_n].index
    d = d[d["genre_top"].isin(keep)]
    order = d.groupby("genre_top")[feature].median().sort_values(ascending=False).index
    groups = [d.loc[d["genre_top"] == g, feature].to_numpy() for g in order]

    # assumption checks
    lev_stat, lev_p = stats.levene(*groups, center="median")
    resid = np.concatenate([g - g.mean() for g in groups])
    sample = resid if len(resid) <= 5000 else np.random.default_rng(0).choice(resid, 5000, replace=False)
    norm_stat, norm_p = stats.normaltest(sample)

    F, p_anova = stats.f_oneway(*groups)
    wF, wdf1, wdf2, p_welch = welch_anova(groups)
    H, p_kw = stats.kruskal(*groups)

    grand = d[feature].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((d[feature] - grand) ** 2).sum()
    eta2 = ss_between / ss_total
    n, k = len(d), len(groups)
    eps2 = (H - k + 1) / (n - k)

    ms_within = (ss_total - ss_between) / (n - k)
    omega2 = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    
    equal_var = lev_p >= ALPHA
    normal = norm_p >= ALPHA
    min_group = min(len(g) for g in groups)
    if equal_var and normal:
        primary = ("One-way ANOVA", F, p_anova)
    elif normal or min_group >= 30:
        primary = ("Welch's ANOVA", wF, p_welch)
    else:
        primary = ("Kruskal-Wallis H", H, p_kw)

    gh = games_howell(d, feature, order)

    pairs = list(itertools.combinations(order, 2))
    mw_p = [stats.mannwhitneyu(d.loc[d.genre_top == a, feature],
                               d.loc[d.genre_top == b, feature]).pvalue for a, b in pairs]
    rej, p_holm, _, _ = multipletests(mw_p, method="holm")
    posthoc = pd.DataFrame({"group1": [a for a, _ in pairs], "group2": [b for _, b in pairs],
                            "median_diff": [d.loc[d.genre_top == a, feature].median()
                                            - d.loc[d.genre_top == b, feature].median() for a, b in pairs],
                            "p_holm": p_holm, "significant": rej})

    summary = d.groupby("genre_top")[feature].agg(["count", "mean", "median", "std"]).loc[order]
    name, stat, p = primary
    sig = p < ALPHA
    top, bottom = order[0], order[-1]
    interp = (
        f"Using {name} as the primary test (chosen from the assumption checks), "
        f"{p_eq(p)}, so we {'reject' if sig else 'fail to reject'} H0 at alpha = {ALPHA}. "
        + (f"{feature.capitalize()} is associated with genre: {top} has the highest median "
           f"({summary.loc[top, 'median']:.2f}) and {bottom} the lowest "
           f"({summary.loc[bottom, 'median']:.2f}). "
                      + (f"The effect size is epsilon-squared = {eps2:.3f} (rank-based; eta-squared = {eta2:.3f}), "
              if name.startswith("Kruskal") else
              f"The effect size is eta-squared = {eta2:.3f} (omega-squared = {omega2:.3f}), ")
           + f"so genre accounts for roughly {100 * (eps2 if name.startswith('Kruskal') else eta2):.1f}% "
           f"of the variation in {feature}. "
           f"Games-Howell post-hoc tests find {int(gh['significant'].sum())} of {len(gh)} genre pairs "
           f"significantly different; the rank-based Kruskal-Wallis test (epsilon-squared = {eps2:.3f}) "
           f"{'agrees' if (p_kw < ALPHA) == sig else 'disagrees'}. "
           if sig else f"There is no evidence that mean {feature} differs across genres. ")
        + "This is an association; it does not show that genre causes the difference."
    )
    return {
        "name": f"{feature.capitalize()} across genres",
        "h0": f"The mean {feature} is the same in every genre.",
        "h1": f"At least one genre has a different mean {feature}.",
        "why": (f"{feature.capitalize()} is continuous and genre is a categorical variable with "
                f"{k} groups, so this is a k-sample comparison of means (one-way ANOVA). Classic ANOVA assumes "
                "normal residuals and equal variances, so both are checked. If variances differ, Welch's ANOVA "
                "is used: it does not assume equal variances, and with at least 30 tracks per genre the central "
                "limit theorem makes it valid for comparing means even when residuals are not normal. "
                "Games-Howell post-hoc tests are used because they also allow unequal variances and group "
                "sizes (Tukey HSD would not). Kruskal-Wallis, with Holm-corrected pairwise Mann-Whitney tests, "
                "is reported as a rank-based robustness check because some genres are strongly skewed."),
        "assumptions": pd.DataFrame({
            "check": ["Levene (equal variances)", "D'Agostino K² (normal residuals)"],
            "statistic": [lev_stat, norm_stat], "p_value": [lev_p, norm_p],
            "assumption_met": [equal_var, normal]}),
        "results": pd.DataFrame({
            "test": ["One-way ANOVA", "Welch's ANOVA", "Kruskal-Wallis H"],
            "statistic": [F, wF, H], "df": [f"{k-1}, {n-k}", f"{wdf1:.0f}, {wdf2:.1f}", f"{k-1}"],
            "p_value": [p_anova, p_welch, p_kw],
            "effect_size": [f"eta² = {eta2:.3f}, omega² = {omega2:.3f}", "", f"epsilon² = {eps2:.3f}"]}),
        "primary": name, "statistic": stat, "p": p, "eta2": eta2, "eps2": eps2, "omega2": omega2,
        "significant": sig, "summary": summary, "games_howell": gh, "posthoc": posthoc, "data": d, "order": list(order),
        "interpretation": interp,
    }


def genre_era_chi2(tracks, top_n=8):
    """Test 2: is genre independent of release era?"""
    d = tracks.dropna(subset=["genre_top", "era"]).copy()
    d["genre_top"] = d["genre_top"].astype(str)
    top = d["genre_top"].value_counts().head(top_n).index
    d = d[d["genre_top"].isin(top)]
    table = pd.crosstab(d["genre_top"], d["era"].astype(str))
    table = table[[c for c in ["Before 2000", "2000-2009", "2010-2017"] if c in table.columns]]
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    pct_low = (expected < 5).mean() * 100
    resid = (table - expected) / np.sqrt(expected)
    sig = p < ALPHA
    hi = resid.stack().idxmax()
    interp = (
        f"chi² = {chi2:.1f} with {dof} degrees of freedom, {p_eq(p)}. We "
        f"{'reject' if sig else 'fail to reject'} H0. "
        + (f"Genre mix is associated with release era (Cramér's V = {cramers_v:.3f}; values below 0.1 are "
           f"usually read as weak). The largest over-representation is {hi[0]} in {hi[1]} "
           f"(standardized residual {resid.loc[hi]:.1f}). " if sig else
           "The genre mix does not differ detectably between eras. ")
        + "Release year is missing for many tracks, so this uses only tracks with a known year."
    )
    return {
        "name": "Genre vs. release era (chi-square test of independence)",
        "h0": "Top-level genre and release era are independent.",
        "h1": "Genre and release era are associated.",
        "why": ("Both variables are categorical (genre and era bin), and we want to know whether the "
                "genre distribution differs across eras. The chi-square test of independence compares "
                f"observed and expected counts. Its assumption (expected counts ≥ 5) holds for "
                f"{100 - pct_low:.0f}% of cells."),
        "table": table, "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
        "residuals": resid, "chi2": chi2, "dof": dof, "p": p, "cramers_v": cramers_v,
        "significant": sig, "interpretation": interp,
    }


def audio_listens_spearman(echo, features):
    """Test 3: Spearman correlation of each audio feature with log listens."""
    d = echo.dropna(subset=["log_listens"])
    rows = []
    for f in features:
        sub = d[[f, "log_listens"]].dropna()
        rho, p = stats.spearmanr(sub[f], sub["log_listens"])
        rows.append({"feature": f, "rho": rho, "p_value": p, "n": len(sub)})
    res = pd.DataFrame(rows)
    res["p_holm"] = multipletests(res["p_value"], method="holm")[1]
    res["significant"] = res["p_holm"] < ALPHA
    res = res.sort_values("rho", key=np.abs, ascending=False)
    strongest = res.iloc[0]
    interp = (
        f"{int(res['significant'].sum())} of {len(res)} features have a significant monotonic "
        f"association with log listens after Holm correction. The strongest is {strongest['feature']} "
        f"(rho = {strongest['rho']:.3f}). |rho| < 0.1 is a negligible relationship, so even "
        "significant correlations here may be too small to matter in practice; with thousands of "
        "tracks, tiny effects become statistically significant."
    )
    return {
        "name": "Audio features vs. popularity (Spearman correlation)",
        "h0": "For each feature, there is no monotonic association with log(listens) (rho = 0).",
        "h1": "rho ≠ 0.",
        "why": ("Listens are heavily right-skewed and the relationship need not be linear, so Spearman's "
                "rank correlation is used instead of Pearson. Eight features are tested at once, so "
                "p-values are Holm-corrected for multiple comparisons."),
        "results": res, "interpretation": interp,
    }