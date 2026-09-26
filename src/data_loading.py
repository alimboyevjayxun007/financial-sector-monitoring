import pandas as pd

from src import config

from pathlib import Path
from typing import Sequence, Union

_SIGNALS_REQUIRED_COLS = [config.ID_COL, "signal_sanasi"]
_TRANSACTIONS_REQUIRED_COLS = [
    config.ID_COL,
    "tranzaksiya_vaqti",
    "kirim_chiqim",
    "tranzaksiya_turi",
    "miqdor_indeksi",
]


def _assert_no_nan(df: pd.DataFrame, cols: Sequence[str], source: str) -> None:
    nan_counts = df[cols].isna().sum()
    bad = nan_counts[nan_counts > 0]
    if not bad.empty:
        raise ValueError(f"{source}: unexpected NaN values found: {bad.to_dict()}")


def load_signals(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signal_sanasi"])
    assert df[config.ID_COL].is_unique, "duplicate signal_id in signals file"
    _assert_no_nan(df, _SIGNALS_REQUIRED_COLS, str(path))
    return df


def load_transactions(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    _assert_no_nan(df, _TRANSACTIONS_REQUIRED_COLS, str(path))
    return df


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("src.data_loading")

    signals = load_signals(config.TRAIN_SIGNALS_PATH)
    transactions = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
    logger.info("Loaded signals: %s, transactions: %s", signals.shape, transactions.shape)
