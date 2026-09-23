"""Track A mission: FeatureBuilder.

TODO(track-a): implement build() so it returns a DataFrame with exactly the
columns listed in config.FEATURE_COLUMNS (+ config.ID_COL, and
config.TARGET_COL when a target column is present in signals_df).

Hard rule: only use transaction rows where
tranzaksiya_vaqti <= signal_sanasi — using later rows is label leakage.
"""
import pandas as pd

from src import config
from src.data_loading import load_signals, load_transactions


def build(signals_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError(
        "Track A: aggregate transactions_df per signal_id into "
        "config.FEATURE_COLUMNS. See ARCHITECTURE.md section 6 for the "
        "exact contract, and config.make_dummy_features() for a stand-in "
        "Track B can already use."
    )


if __name__ == "__main__":
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_signals = load_signals(config.TRAIN_SIGNALS_PATH)
    train_transactions = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
    train_features = build(train_signals, train_transactions)
    train_features.to_parquet(config.TRAIN_FEATURES_PATH, index=False)

    test_signals = load_signals(config.TEST_SIGNALS_PATH)
    test_transactions = load_transactions(config.TEST_TRANSACTIONS_PATH)
    test_features = build(test_signals, test_transactions)
    test_features.to_parquet(config.TEST_FEATURES_PATH, index=False)

    print(f"wrote {config.TRAIN_FEATURES_PATH} and {config.TEST_FEATURES_PATH}")
