"""Export every report figure as a PNG into figures/.

Usage (from repo root, after download_data.py and train_models.py):
    python src/make_figures.py
"""
import joblib
import matplotlib.pyplot as plt

import cleaning
import eda
from config import ECHONEST_AUDIO, FIGURES, MODEL_RESULTS
from data_loader import (feature_columns, load_cleaning_log, load_echonest,
                         load_missing_summary, load_small_features, load_tracks)


def main():
    FIGURES.mkdir(exist_ok=True)
    tracks, small, echo = load_tracks(), load_small_features(), load_echonest()
    miss, log = load_missing_summary(), load_cleaning_log()
    fcols = feature_columns(small)
    figs = {
        "01_missing_values": eda.fig_missing(miss),
        "02_genre_counts": eda.fig_genre_counts(tracks),
        "03_duration": eda.fig_duration(tracks, log),
        "04_listens": eda.fig_listens(tracks),
        "05_years": eda.fig_years(tracks),
        "06_uploads_by_genre": eda.fig_uploads_by_genre(tracks),
        "07_echonest_correlation": eda.fig_corr(echo, ECHONEST_AUDIO),
        "08_pca": eda.fig_pca(small, fcols),
        "09_feature_outliers": eda.fig_feature_outliers(cleaning.feature_outlier_summary(small[fcols])),
    }
    for name, (fig, caption, _) in figs.items():
        fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"saved figures/{name}.png  -  {caption}")


if __name__ == "__main__":
    main()