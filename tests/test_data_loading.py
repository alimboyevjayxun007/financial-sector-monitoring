import pandas as pd
import pytest

from src import config
from src.data_loading import load_signals, load_transactions


def test_load_signals_parses_dates_and_checks_uniqueness(tmp_path):
    csv_path = tmp_path / "signals.csv"
    pd.DataFrame(
        {
            "signal_id": ["SG_1", "SG_2"],
            "signal_sanasi": ["2025-06-01", "2025-06-02"],
            "eskalatsiya": [0, 1],
        }
    ).to_csv(csv_path, index=False)

    df = load_signals(csv_path)
    assert pd.api.types.is_datetime64_any_dtype(df["signal_sanasi"])
    assert list(df[config.ID_COL]) == ["SG_1", "SG_2"]


def test_load_signals_rejects_duplicate_ids(tmp_path):
    csv_path = tmp_path / "signals_dup.csv"
    pd.DataFrame(
        {
            "signal_id": ["SG_1", "SG_1"],
            "signal_sanasi": ["2025-06-01", "2025-06-02"],
            "eskalatsiya": [0, 1],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(AssertionError):
        load_signals(csv_path)


def test_load_signals_rejects_nan_signal_sanasi(tmp_path):
    csv_path = tmp_path / "signals_nan.csv"
    pd.DataFrame(
        {
            "signal_id": ["SG_1", "SG_2"],
            "signal_sanasi": ["2025-06-01", None],
            "eskalatsiya": [0, 1],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="NaN"):
        load_signals(csv_path)


def test_load_transactions_rejects_nan_in_required_columns(tmp_path):
    parquet_path = tmp_path / "txn_nan.parquet"
    pd.DataFrame(
        {
            "signal_id": ["SG_1", "SG_2"],
            "tranzaksiya_vaqti": pd.to_datetime(["2025-06-01", "2025-06-02"]),
            "kirim_chiqim": ["kirim", None],
            "tranzaksiya_turi": ["karta", "naqd"],
            "miqdor_indeksi": [0.5, -0.3],
        }
    ).to_parquet(parquet_path, index=False)

    with pytest.raises(ValueError, match="NaN"):
        load_transactions(parquet_path)


def test_real_train_and_test_signals_have_no_overlap_if_available():
    if not config.TRAIN_SIGNALS_PATH.exists():
        pytest.skip("fintech_track_data/ not present (gitignored, not included in the repo)")
    train_signals = load_signals(config.TRAIN_SIGNALS_PATH)
    test_signals = load_signals(config.TEST_SIGNALS_PATH)
    overlap = set(train_signals[config.ID_COL]) & set(test_signals[config.ID_COL])
    assert len(overlap) == 0


def test_real_transactions_load_if_available():
    if not config.TRAIN_TRANSACTIONS_PATH.exists():
        pytest.skip("fintech_track_data/ not present (gitignored, not included in the repo)")
    transactions = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
    expected_cols = {
        "signal_id",
        "tranzaksiya_vaqti",
        "kirim_chiqim",
        "tranzaksiya_turi",
        "miqdor_indeksi",
    }
    assert expected_cols.issubset(set(transactions.columns))
