import json

import numpy as np
import pandas as pd

import cleaning
from config import (CLEANING_LOG, DATA_PROCESSED, DATA_RAW, ECHONEST_AUDIO,
                    ECHONEST_PQ, MISSING_CSV, SMALL_FEATURES_PQ, TRACKS_PQ)

DATE_COLS = [("track", "date_created"), ("track", "date_recorded"),
             ("album", "date_created"), ("album", "date_released"),
             ("artist", "date_created"), ("artist", "active_year_begin"),
             ("artist", "active_year_end")]

def load_raw_tracks(path=None):
    """tracks.csv has a 2-row header: (group, field)."""
    tracks = pd.read_csv(path or DATA_RAW / "tracks.csv", index_col=0,
                         header=[0, 1], low_memory=False)
    for col in DATE_COLS:
        if col in tracks.columns:
            tracks[col] = pd.to_datetime(tracks[col], errors="coerce")
    return tracks

def load_raw_features(path=None):
    """features.csv has a 3-row header: (feature, statistic, coefficient)."""
    return pd.read_csv(path or DATA_RAW / "features.csv", index_col=0,
                       header=[0, 1, 2])

def load_raw_echonest(path=None):
    """echonest.csv has a 3-row header: ('echonest', group, field)."""
    return pd.read_csv(path or DATA_RAW / "echonest.csv", index_col=0,
                       header=[0, 1, 2])

def flatten_columns(df, sep="_"):
    """('mfcc', 'mean', '01') -> 'mfcc_mean_01'."""
    out = df.copy()
    out.columns = [sep.join(str(p) for p in col if str(p) != "") for col in df.columns]
    return out

def build_processed():
    """Raw CSVs -> data/processed/*.parquet + missingness + cleaning log."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    log = {}

    print("Loading tracks.csv...")
    raw = load_raw_tracks()
    log["raw_tracks_shape"] = list(raw.shape)
    log["duplicate_track_ids"] = int(raw.index.duplicated().sum())

    # 1) missingness on ALL 52 raw columns, before anything is dropped
    miss = cleaning.missing_summary(raw)
    miss.to_csv(MISSING_CSV, index=False)

    # 2) keep the analysis columns, then clean them
    tracks = raw[list(TRACK_KEEP)].copy()
    tracks.columns = list(TRACK_KEEP.values())
    tracks, track_log = cleaning.clean_tracks(tracks)
    log.update(track_log)
    tracks.to_parquet(TRACKS_PQ)
    print(f"  wrote {TRACKS_PQ.name}: {tracks.shape}")

    # 3) features for the balanced 'small' subset (classification data)
    print("Loading features.csv...")
    feats = flatten_columns(load_raw_features()).astype(np.float32)
    log["raw_features_shape"] = list(feats.shape)
    log["features_missing_cells"] = int(feats.isna().sum().sum())
    small_ids = tracks.index[(tracks["subset"] == "small") & tracks["genre_top"].notna()]
    small = feats.loc[feats.index.intersection(small_ids)].copy()
    small.insert(0, "split", tracks.loc[small.index, "split"].astype(str))
    small.insert(0, "genre_top", tracks.loc[small.index, "genre_top"].astype(str))
    log["small_subset_rows"] = int(len(small))
    small.to_parquet(SMALL_FEATURES_PQ)
    print(f"  wrote {SMALL_FEATURES_PQ.name}: {small.shape}")

    # 4) echonest audio descriptors (energy etc.) joined to genre + listens
    print("Loading echonest.csv...")
    echo = load_raw_echonest()
    audio = echo[("echonest", "audio_features")].copy()
    audio.columns = [str(c) for c in audio.columns]
    audio = audio[ECHONEST_AUDIO].astype(float)
    audio = audio.join(tracks[["genre_top", "listens", "log_listens", "subset"]], how="left")
    log["echonest_rows"] = int(len(audio))
    log["echonest_rows_with_genre"] = int(audio["genre_top"].notna().sum())
    audio.to_parquet(ECHONEST_PQ)
    print(f"  wrote {ECHONEST_PQ.name}: {audio.shape}")

    CLEANING_LOG.write_text(json.dumps(log, indent=2, default=str))
    print("Done. Cleaning log:", CLEANING_LOG)
    return log

TRACK_KEEP = {
    ("track", "genre_top"): "genre_top",
    ("track", "duration"): "duration",
    ("track", "listens"): "listens",
    ("track", "favorites"): "favorites",
    ("track", "interest"): "interest",
    ("track", "bit_rate"): "bit_rate",
    ("track", "license"): "license",
    ("track", "language_code"): "language_code",
    ("track", "date_created"): "date_created",
    ("track", "date_recorded"): "date_recorded",
    ("album", "date_released"): "album_date_released",
    ("album", "type"): "album_type",
    ("album", "listens"): "album_listens",
    ("artist", "id"): "artist_id",
    ("artist", "location"): "artist_location",
    ("set", "split"): "split",
    ("set", "subset"): "subset",
}

def load_tracks():
    return pd.read_parquet(TRACKS_PQ)


def load_small_features():
    return pd.read_parquet(SMALL_FEATURES_PQ)


def load_echonest():
    return pd.read_parquet(ECHONEST_PQ)


def load_missing_summary():
    return pd.read_csv(MISSING_CSV)


def load_cleaning_log():
    return json.loads(CLEANING_LOG.read_text())


def processed_available():
    return all(p.exists() for p in [TRACKS_PQ, SMALL_FEATURES_PQ, ECHONEST_PQ,
                                    MISSING_CSV, CLEANING_LOG])


def feature_columns(df):
    return [c for c in df.columns if c not in ("genre_top", "split")]


if __name__ == "__main__":
    build_processed()