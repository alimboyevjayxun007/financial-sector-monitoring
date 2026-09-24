"""Shared paths and the feature contract between Track A (features) and Track B (model)."""
from pathlib import Path

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
    "hour_entropy",
    "hour_maxshare",
    "dow_entropy",
    "dow_maxshare",
]
