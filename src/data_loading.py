"""Track A: DataLoader. Reads the raw CSV/Parquet files as-is."""
import pandas as pd

from src import config


def load_signals(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signal_sanasi"])
    assert df[config.ID_COL].is_unique, "duplicate signal_id in signals file"
    return df


def load_transactions(path) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    signals = load_signals(config.TRAIN_SIGNALS_PATH)
    transactions = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
    print(f"signals: {signals.shape}, transactions: {transactions.shape}")
