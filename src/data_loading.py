"""Track A: DataLoader. Reads the raw CSV/Parquet files as-is.

Fails fast on NaN in critical columns rather than silently letting them
flow into features.py, where a NaN in e.g. tranzaksiya_turi would silently
drop that row from every frac_* category (making them not sum to 1) with
no error or warning -- see README.md production-readiness notes.
"""
import pandas as pd

from src import config

_SIGNALS_REQUIRED_COLS = [config.ID_COL, "signal_sanasi"]
_TRANSACTIONS_REQUIRED_COLS = [
    config.ID_COL,
    "tranzaksiya_vaqti",
    "kirim_chiqim",
    "tranzaksiya_turi",
    "miqdor_indeksi",
]


def _assert_no_nan(df: pd.DataFrame, cols: list, source: str) -> None:
    nan_counts = df[cols].isna().sum()
    bad = nan_counts[nan_counts > 0]
    if not bad.empty:
        raise ValueError(f"{source}: unexpected NaN values found: {bad.to_dict()}")


def load_signals(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signal_sanasi"])
    assert df[config.ID_COL].is_unique, "duplicate signal_id in signals file"
    _assert_no_nan(df, _SIGNALS_REQUIRED_COLS, str(path))
    return df


def load_transactions(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    _assert_no_nan(df, _TRANSACTIONS_REQUIRED_COLS, str(path))
    return df


if __name__ == "__main__":
    signals = load_signals(config.TRAIN_SIGNALS_PATH)
    transactions = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
    print(f"signals: {signals.shape}, transactions: {transactions.shape}")
