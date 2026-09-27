import numpy as np
import pandas as pd

from src.features import FEATURE_FAMILIES, build_features, prepare_transactions


def _signals():
    return pd.DataFrame(
        {
            "signal_id": ["SG_1", "SG_2", "SG_3"],
            "signal_sanasi": pd.to_datetime(["2025-03-10", "2025-03-10", "2025-03-10"]),
        }
    )


def _transactions():
    return pd.DataFrame(
        {
            "signal_id": ["SG_1", "SG_1", "SG_2"],
            "tranzaksiya_vaqti": pd.to_datetime(
                ["2025-03-01 10:00", "2025-03-05 23:30", "2025-03-09 04:00"]
            ),
            "kirim_chiqim": ["kirim", "chiqim", "kirim"],
            "tranzaksiya_turi": ["karta", "karta", "naqd"],
            "miqdor_indeksi": [1.0, -0.5, 2.0],
        }
    )


def test_future_transactions_are_dropped():
    signals = _signals()
    tx = _transactions()
    tx.loc[0, "tranzaksiya_vaqti"] = pd.Timestamp("2025-03-12 10:00")
    prepared = prepare_transactions(tx, signals)
    assert (prepared["days_before"] >= 0).all()
    assert len(prepared) == 2


def test_column_schema_is_identical_when_a_category_is_absent():
    signals = _signals()
    full = _transactions()
    narrow = full[full["tranzaksiya_turi"] == "karta"].copy()

    wide_cols = build_features(prepare_transactions(full, signals), signals).columns
    narrow_cols = build_features(prepare_transactions(narrow, signals), signals).columns

    assert list(wide_cols) == list(narrow_cols)
    assert any("xalqaro" in c for c in wide_cols)


def test_every_signal_gets_exactly_one_row_including_signals_with_no_transactions():
    signals = _signals()
    result = build_features(prepare_transactions(_transactions(), signals), signals)

    assert list(result.index) == ["SG_1", "SG_2", "SG_3"]
    assert not result.index.duplicated().any()
    assert result.loc["SG_3"].isna().any() or (result.loc["SG_3"] == 0).any()


def test_count_columns_are_zero_not_nan_for_signals_with_no_transactions():
    signals = _signals()
    result = build_features(prepare_transactions(_transactions(), signals), signals)
    assert result.loc["SG_3", "base_cnt"] == 0


def test_all_families_are_registered_and_produce_columns():
    signals = _signals()
    prepared = prepare_transactions(_transactions(), signals)
    for name in FEATURE_FAMILIES:
        frame = build_features(prepared, signals, families=[name])
        assert frame.shape[1] > 0, name
        assert list(frame.index) == ["SG_1", "SG_2", "SG_3"], name


def test_duplicate_signal_ids_do_not_multiply_aggregates():
    """A signal listed twice must not double its own transactions."""
    signals = pd.concat([_signals(), _signals().iloc[[0]]], ignore_index=True)
    prepared = prepare_transactions(_transactions(), signals)
    result = build_features(prepared, signals)

    assert not result.index.duplicated().any()
    assert result.loc["SG_1", "base_cnt"] == 2


def test_zero_transaction_signals_get_nan_not_zero_for_amount_columns():
    """0.0 is a reachable real sum, so it must not stand for 'no activity'."""
    signals = _signals()
    result = build_features(prepare_transactions(_transactions(), signals), signals)
    row = result.loc["SG_3"]

    assert row["base_cnt"] == 0
    assert row["dir_n_kirim"] == 0
    for column in ("dir_s_kirim", "dir_s_chiqim", "ty_s_karta", "base_sum", "flow_net"):
        assert pd.isna(row[column]), column


def test_signal_date_features_are_computed_against_a_fixed_reference():
    """sig_tnum and sig_same_day_n must not depend on which split they are in."""
    train = _signals()
    test = train.iloc[[0]].copy()
    test["signal_id"] = ["SG_9"]

    train_feat = build_features(
        prepare_transactions(_transactions(), train), train, families=["signal_date"]
    )
    test_feat = build_features(
        prepare_transactions(_transactions().iloc[:0], test), test,
        families=["signal_date"], reference=train,
    )

    assert test_feat.loc["SG_9", "sig_tnum"] == train_feat.loc["SG_1", "sig_tnum"]
    assert test_feat.loc["SG_9", "sig_same_day_n"] == train_feat.loc["SG_1", "sig_same_day_n"]
