import pandas as pd
import pytest

from src import config
from src.features import build


def _signals(rows):
    df = pd.DataFrame(rows, columns=["signal_id", "signal_sanasi", "eskalatsiya"])
    df["signal_sanasi"] = pd.to_datetime(df["signal_sanasi"])
    return df


def _transactions(rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "signal_id",
            "tranzaksiya_vaqti",
            "kirim_chiqim",
            "tranzaksiya_turi",
            "miqdor_indeksi",
        ],
    )
    df["tranzaksiya_vaqti"] = pd.to_datetime(df["tranzaksiya_vaqti"])
    return df


def test_output_columns_match_contract():
    signals = _signals([("SG_1", "2025-06-01", 0), ("SG_2", "2025-06-01", 1)])
    transactions = _transactions(
        [
            ("SG_1", "2025-05-30", "kirim", "karta", 0.5),
            ("SG_2", "2025-05-30", "chiqim", "naqd", -0.3),
        ]
    )
    out = build(signals, transactions)
    assert list(out.columns) == [config.ID_COL] + config.FEATURE_COLUMNS + [config.TARGET_COL]


def test_one_row_per_signal_even_with_no_transactions():
    signals = _signals(
        [("SG_1", "2025-06-01", 0), ("SG_2", "2025-06-01", 1), ("SG_3", "2025-06-01", 0)]
    )
    # SG_3 has no transactions at all
    transactions = _transactions(
        [
            ("SG_1", "2025-05-30", "kirim", "karta", 0.5),
            ("SG_2", "2025-05-30", "chiqim", "naqd", -0.3),
        ]
    )
    out = build(signals, transactions)
    assert len(out) == 3
    assert set(out[config.ID_COL]) == {"SG_1", "SG_2", "SG_3"}
    sg3 = out[out[config.ID_COL] == "SG_3"].iloc[0]
    assert sg3["n_txn"] == 0
    assert not sg3[config.FEATURE_COLUMNS].isna().any()


def test_no_missing_values_anywhere():
    signals = _signals([("SG_1", "2025-06-01", 0), ("SG_2", "2025-06-01", 1)])
    transactions = _transactions(
        [
            # SG_1 has a single transaction -> std of one value would be NaN
            ("SG_1", "2025-05-30", "kirim", "karta", 0.5),
            ("SG_2", "2025-05-29", "chiqim", "naqd", -0.3),
            ("SG_2", "2025-05-30", "kirim", "bank_otkazmasi", 1.1),
        ]
    )
    out = build(signals, transactions)
    assert out.isna().sum().sum() == 0


def test_future_transactions_are_excluded_no_leakage():
    signals = _signals([("SG_1", "2025-06-01", 0)])
    transactions = _transactions(
        [
            ("SG_1", "2025-05-30", "kirim", "karta", 0.5),  # before signal: kept
            ("SG_1", "2025-06-02", "kirim", "xalqaro", 9.0),  # after signal: must be dropped
        ]
    )
    out = build(signals, transactions).iloc[0]
    assert out["n_txn"] == 1
    assert out["frac_xalqaro"] == 0.0
    assert out["amt_max"] == 0.5


def test_fractions_are_within_unit_range():
    signals = _signals([("SG_1", "2025-06-01", 0)])
    transactions = _transactions(
        [
            ("SG_1", "2025-05-01", "kirim", "karta", 0.1),
            ("SG_1", "2025-05-02", "chiqim", "naqd", -3.0),
            ("SG_1", "2025-05-31", "kirim", "xalqaro", 2.5),
        ]
    )
    out = build(signals, transactions)
    frac_cols = [c for c in config.FEATURE_COLUMNS if c.startswith("frac_")]
    for col in frac_cols:
        assert out[col].between(0, 1).all(), col


def test_target_column_omitted_when_absent_from_signals():
    signals = pd.DataFrame(
        {"signal_id": ["SG_1"], "signal_sanasi": pd.to_datetime(["2025-06-01"])}
    )
    transactions = _transactions([("SG_1", "2025-05-30", "kirim", "karta", 0.5)])
    out = build(signals, transactions)
    assert config.TARGET_COL not in out.columns
    assert list(out.columns) == [config.ID_COL] + config.FEATURE_COLUMNS


def test_real_data_end_to_end_if_available():
    if not config.TRAIN_SIGNALS_PATH.exists():
        pytest.skip("fintech_track_data/ not present (gitignored, not included in the repo)")
    from src.data_loading import load_signals, load_transactions

    signals = load_signals(config.TRAIN_SIGNALS_PATH)
    signals = signals.head(200)
    transactions = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
    transactions = transactions[transactions[config.ID_COL].isin(signals[config.ID_COL])]

    out = build(signals, transactions)
    assert len(out) == len(signals)
    assert out.isna().sum().sum() == 0
    assert list(out.columns) == [config.ID_COL] + config.FEATURE_COLUMNS + [config.TARGET_COL]
