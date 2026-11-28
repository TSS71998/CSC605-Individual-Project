from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
AI_LOG = ROOT / "ai_log"

FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
DATASET_PAGE = "https://github.com/mdeff/fma"

GITHUB_REPO = "https://github.com/TSS71998/CSC605-Individual-Project"

TRACKS_PQ = DATA_PROCESSED / "tracks_clean.parquet"
MISSING_CSV = DATA_PROCESSED / "missing_summary.csv"
SMALL_FEATURES_PQ = DATA_PROCESSED / "small_features.parquet"
ECHONEST_PQ = DATA_PROCESSED / "echonest_audio.parquet"
CLEANING_LOG = DATA_PROCESSED / "cleaning_log.json"
MODEL_RESULTS = RESULTS / "model_results.joblib"

RANDOM_STATE = 42

ECHONEST_AUDIO = ["acousticness", "danceability", "energy", "instrumentalness",
                  "liveness", "speechiness", "tempo", "valence"]

FEATURE_FAMILIES = ["chroma_cens", "chroma_cqt", "chroma_stft", "mfcc", "rmse",
                    "spectral_bandwidth", "spectral_centroid", "spectral_contrast",
                    "spectral_rolloff", "tonnetz", "zcr"]