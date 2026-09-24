"""Track B: training entry point.

Falls back to config.make_dummy_features() when data/processed/train_features.parquet
does not exist yet, so this script is runnable from day one — swap in the real
file once Track A publishes it, no code change needed here.
"""
import pandas as pd

from src import config
from src.data_loading import load_signals
from src.model import cross_validate, train


def load_train_features() -> pd.DataFrame:
    if config.TRAIN_FEATURES_PATH.exists():
        # features.py already bakes TARGET_COL into this file (see build()),
        # so it's read as-is — no re-merge needed.
        return pd.read_parquet(config.TRAIN_FEATURES_PATH)
    print(f"[train] {config.TRAIN_FEATURES_PATH} not found yet, using dummy features")
    signals = load_signals(config.TRAIN_SIGNALS_PATH)
    features = config.make_dummy_features(signals)
    features[config.TARGET_COL] = signals[config.TARGET_COL].values
    return features


if __name__ == "__main__":
    df = load_train_features()
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COL]

    auc = cross_validate(X, y)
    print(f"CV ROC-AUC: {auc:.4f}")

    model = train(X, y)
    import joblib

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.OUTPUTS_DIR / "model.pkl")
    print(f"model saved to {config.OUTPUTS_DIR / 'model.pkl'}")
