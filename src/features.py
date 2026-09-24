"""Track A: FeatureBuilder.

Aggregates each signal_id's transaction history (as of its signal_sanasi)
into the fixed set of columns in config.FEATURE_COLUMNS — see
ARCHITECTURE.md section 6 for the contract.

Hard rule: only tranzaksiya_vaqti <= signal_sanasi rows are used. Transaction
timestamps run past 2026-12-31 while signals start in 2025, so some
transaction rows for a given signal_id occur AFTER that signal's date —
including them would leak the future into the features.
"""
import pandas as pd

from src import config
from src.data_loading import load_signals, load_transactions

_TXN_TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
_NIGHT_HOURS = set(range(0, 6))  # 00:00–05:59
_EXTREME_THRESHOLD = 2.0
_RECENCY_WINDOWS = {"n_txn_1d": 1, "n_txn_7d": 7, "n_txn_30d": 30}


def build(signals_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    df = transactions_df.merge(
        signals_df[[config.ID_COL, "signal_sanasi"]], on=config.ID_COL, how="inner"
    )
    df = df[df["tranzaksiya_vaqti"] <= df["signal_sanasi"]].copy()

    df["days_before"] = (
        df["signal_sanasi"] - df["tranzaksiya_vaqti"]
    ).dt.total_seconds() / 86400
    df["is_night"] = df["tranzaksiya_vaqti"].dt.hour.isin(_NIGHT_HOURS).astype(int)
    df["is_weekend"] = df["tranzaksiya_vaqti"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_extreme"] = (df["miqdor_indeksi"].abs() > _EXTREME_THRESHOLD).astype(int)
    df["is_kirim"] = (df["kirim_chiqim"] == "kirim").astype(int)
    for t in _TXN_TYPES:
        df[f"is_{t}"] = (df["tranzaksiya_turi"] == t).astype(int)

    agg = df.groupby(config.ID_COL).agg(
        n_txn=("miqdor_indeksi", "count"),
        amt_mean=("miqdor_indeksi", "mean"),
        amt_std=("miqdor_indeksi", "std"),
        amt_max=("miqdor_indeksi", "max"),
        amt_sum=("miqdor_indeksi", "sum"),
        frac_kirim=("is_kirim", "mean"),
        frac_karta=("is_karta", "mean"),
        frac_bank_otkazmasi=("is_bank_otkazmasi", "mean"),
        frac_naqd=("is_naqd", "mean"),
        frac_xalqaro=("is_xalqaro", "mean"),
        frac_night=("is_night", "mean"),
        frac_weekend=("is_weekend", "mean"),
        frac_extreme=("is_extreme", "mean"),
        span_days=("tranzaksiya_vaqti", lambda s: (s.max() - s.min()).total_seconds() / 86400),
    ).reset_index()

    for col, window in _RECENCY_WINDOWS.items():
        recent = (
            df[df["days_before"] <= window]
            .groupby(config.ID_COL)
            .size()
            .rename(col)
        )
        agg = agg.merge(recent, on=config.ID_COL, how="left")

    # Signals with zero prior transactions never appear in `agg` at all —
    # reindex to every signal_id so the output always has exactly one row
    # per input signal, with all-zero features for those cases.
    full = signals_df[[config.ID_COL]].merge(agg, on=config.ID_COL, how="left")
    numeric_cols = [c for c in full.columns if c != config.ID_COL]
    full[numeric_cols] = full[numeric_cols].fillna(0.0)

    full["velocity"] = full["n_txn"] / (full["span_days"] + 1)

    if config.TARGET_COL in signals_df.columns:
        full = full.merge(
            signals_df[[config.ID_COL, config.TARGET_COL]], on=config.ID_COL
        )
        out_cols = [config.ID_COL] + config.FEATURE_COLUMNS + [config.TARGET_COL]
    else:
        out_cols = [config.ID_COL] + config.FEATURE_COLUMNS

    return full[out_cols].reset_index(drop=True)


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

    print(f"wrote {config.TRAIN_FEATURES_PATH} ({train_features.shape})")
    print(f"wrote {config.TEST_FEATURES_PATH} ({test_features.shape})")
