"""Genre classification on the 518 pre-computed audio features (fma_small).

  Baseline : Logistic Regression (multinomial, standardized)  + k-NN for reference
  Advanced : Random Forest, Histogram Gradient Boosting

Protocol: hyperparameters are chosen on the official VALIDATION split by
macro-F1; the chosen model is scored once on the TEST split.

Run `python src/train_models.py` to precompute results/model_results.joblib,
so the Streamlit app does not retrain on every load.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             top_k_accuracy_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from cleaning import feature_family
from config import RANDOM_STATE
from data_loader import feature_columns


def model_grid():
    """name -> (role, list of (param_label, estimator factory))."""
    return {
        "Logistic Regression": ("Baseline", [
            (f"C={c}", lambda c=c: make_pipeline(StandardScaler(), LogisticRegression(
                C=c, max_iter=3000, random_state=RANDOM_STATE))) for c in [0.01, 0.1, 1.0]]),
        "k-NN": ("Baseline (reference)", [
            (f"k={k}", lambda k=k: make_pipeline(StandardScaler(), KNeighborsClassifier(
                n_neighbors=k, weights="distance"))) for k in [5, 15, 31]]),
        "Random Forest": ("Advanced", [
            (f"min_samples_leaf={m}", lambda m=m: RandomForestClassifier(
                n_estimators=500, min_samples_leaf=m, max_features="sqrt", n_jobs=-1,
                random_state=RANDOM_STATE)) for m in [1, 3]]),
        "Gradient Boosting": ("Advanced", [
            (f"lr={lr}", lambda lr=lr: HistGradientBoostingClassifier(
                learning_rate=lr, max_iter=400, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20, l2_regularization=1.0, random_state=RANDOM_STATE))
            for lr in [0.05, 0.1]]),
    }


def split_data(small):
    X_cols = feature_columns(small)
    le = LabelEncoder().fit(small["genre_top"])
    parts = {}
    for name in ["training", "validation", "test"]:
        d = small[small["split"] == name]
        parts[name] = (d[X_cols], le.transform(d["genre_top"]))
    return parts, le, X_cols


def score(y, pred, proba):
    return {"accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "top2_accuracy": top_k_accuracy_score(y, proba, k=2, labels=np.arange(proba.shape[1]))}


def grouped_permutation_importance(model, X, y, n_repeats=3, seed=RANDOM_STATE):
    """Drop in macro-F1 when all columns of one feature family are shuffled together."""
    rng = np.random.default_rng(seed)
    base = f1_score(y, model.predict(X), average="macro")
    fams = pd.Series([feature_family(c) for c in X.columns], index=X.columns)
    rows = []
    for fam, cols in fams.groupby(fams).groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[list(cols)] = Xp[list(cols)].to_numpy()[perm]
            drops.append(base - f1_score(y, model.predict(Xp), average="macro"))
        rows.append({"family": fam, "n_features": len(cols),
                     "importance_mean": np.mean(drops), "importance_std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def train_all(small, verbose=True):
    parts, le, X_cols = split_data(small)
    Xtr, ytr = parts["training"]
    Xva, yva = parts["validation"]
    Xte, yte = parts["test"]
    classes = list(le.classes_)
    out = {"classes": classes, "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
           "n_features": len(X_cols), "tuning": [], "models": {}}

    for name, (role, grid) in model_grid().items():
        best = None
        for label, factory in grid:
            t0 = time.time()
            m = factory().fit(Xtr, ytr)
            s = score(yva, m.predict(Xva), m.predict_proba(Xva))
            out["tuning"].append({"model": name, "params": label, "val_macro_f1": s["macro_f1"],
                                  "val_accuracy": s["accuracy"], "fit_seconds": time.time() - t0})
            if verbose:
                print(f"  {name:20s} {label:22s} val macro-F1 = {s['macro_f1']:.3f}")
            if best is None or s["macro_f1"] > best[0]:
                best = (s["macro_f1"], label, m)
        _, label, m = best
        pred, proba = m.predict(Xte), m.predict_proba(Xte)
        out["models"][name] = {
            "role": role, "best_params": label, "test": score(yte, pred, proba),
            "confusion": confusion_matrix(yte, pred, labels=np.arange(len(classes))),
            "report": pd.DataFrame(classification_report(
                yte, pred, labels=np.arange(len(classes)), target_names=classes,
                output_dict=True, zero_division=0)).T,
            "_estimator": m,
        }

    best_name = max(out["models"], key=lambda k: out["models"][k]["test"]["macro_f1"])
    out["best_model"] = best_name
    if verbose:
        print(f"Permutation importance for {best_name} ...")
    out["importance"] = grouped_permutation_importance(
        out["models"][best_name]["_estimator"], Xte, yte)
    for v in out["models"].values():
        v.pop("_estimator")  # keep the results file small
    out["tuning"] = pd.DataFrame(out["tuning"])
    return out


def comparison_table(results):
    rows = []
    for name, r in results["models"].items():
        rows.append({"model": name, "role": r["role"], "chosen params": r["best_params"],
                     **{k: round(v, 3) for k, v in r["test"].items()}})
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def top_confusions(results, model_name, n=6):
    cm = np.asarray(results["models"][model_name]["confusion"], float)
    classes = results["classes"]
    rates = cm / cm.sum(axis=1, keepdims=True)
    rows = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j:
                rows.append({"true genre": classes[i], "predicted as": classes[j],
                             "share of true genre": rates[i, j], "count": int(cm[i, j])})
    return pd.DataFrame(rows).sort_values("share of true genre", ascending=False).head(n)