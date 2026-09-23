"""Shared paths and the feature contract between Track A (features) and Track B (model).

Track B can start immediately with make_dummy_features() while Track A builds
the real pipeline in features.py — both produce a DataFrame with the same
columns, so train.py / predict.py never need to change.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT / "fintech_track_data" / "fintech_data"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

TRAIN_SIGNALS_PATH = RAW_DATA_DIR / "train_signals.csv"
TRAIN_TRANSACTIONS_PATH = RAW_DATA_DIR / "train_transactions.parquet"
TEST_SIGNALS_PATH = RAW_DATA_DIR / "test_signals.csv"
TEST_TRANSACTIONS_PATH = RAW_DATA_DIR / "test_transactions.parquet"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.parquet"
TEST_FEATURES_PATH = PROCESSED_DIR / "test_features.parquet"

ID_COL = "signal_id"
TARGET_COL = "eskalatsiya"

# The agreed contract (see ARCHITECTURE.md section 6). Track A must produce
# exactly these columns (plus TARGET_COL for the train split); Track B must
# only ever read these columns by name.
FEATURE_COLUMNS = [
    "n_txn",
    "amt_mean",
    "amt_std",
    "amt_max",
    "amt_sum",
    "frac_kirim",
    "frac_karta",
    "frac_bank_otkazmasi",
    "frac_naqd",
    "frac_xalqaro",
    "frac_night",
    "frac_weekend",
    "frac_extreme",
    "n_txn_1d",
    "n_txn_7d",
    "n_txn_30d",
    "span_days",
    "velocity",
]


def make_dummy_features(signals_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Stand-in for FeatureBuilder.build(), so Track B can build/test the
    modeling pipeline before Track A's real feature table exists. Replace
    the call site with the real data_loading + features pipeline once
    data/processed/*.parquet is available.
    """
    rng = np.random.default_rng(seed)
    n = len(signals_df)
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    df.insert(0, ID_COL, signals_df[ID_COL].values)
    return df
