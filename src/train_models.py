"""Train and evaluate all models, then save results for the Streamlit app.

Usage (from repo root):  python src/train_models.py
"""
import joblib

from config import MODEL_RESULTS, RESULTS
from data_loader import load_small_features
from models import comparison_table, train_all


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    small = load_small_features()
    print(f"Training on fma_small: {small.shape[0]} tracks, {small.shape[1] - 2} features")
    results = train_all(small)
    joblib.dump(results, MODEL_RESULTS, compress=3)
    print(comparison_table(results).to_string(index=False))
    print("Saved", MODEL_RESULTS)


if __name__ == "__main__":
    main()