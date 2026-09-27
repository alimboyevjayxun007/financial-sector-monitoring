"""End-to-end cover for the prediction path, which never ran against real test
data because the organizer-supplied test parquet is corrupt."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import CorruptParquetError
from src.submission import SubmissionError


def _write_fixture(root, n_train=300, n_test=120, seed=0, silent_test_alerts=3):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)

    def make(prefix, n, start, labelled):
        ids = [f"{prefix}_{i:06d}" for i in range(start, start + n)]
        dates = pd.to_datetime("2025-06-01") + pd.to_timedelta(rng.integers(0, 300, n), "D")
        frame = pd.DataFrame({"signal_id": ids, "signal_sanasi": dates.strftime("%Y-%m-%d")})
        if labelled:
            frame["eskalatsiya"] = rng.binomial(1, 0.17, n)
        return frame, ids, dates

    train, train_ids, train_dates = make("SG", n_train, 0, True)
    test, test_ids, test_dates = make("SG", n_test, n_train, False)

    def transactions(ids, dates, skip_last):
        rows = []
        for sid, date in zip(ids[: len(ids) - skip_last], dates[: len(dates) - skip_last]):
            k = int(rng.integers(5, 40))
            offsets = rng.uniform(0, 180, k)
            rows.append(
                pd.DataFrame({
                    "signal_id": sid,
                    "tranzaksiya_vaqti": date - pd.to_timedelta(offsets, "D"),
                    "kirim_chiqim": rng.choice(config.DIRECTIONS, k),
                    "tranzaksiya_turi": rng.choice(config.TX_TYPES, k),
                    "miqdor_indeksi": rng.normal(size=k),
                })
            )
        return pd.concat(rows, ignore_index=True)

    train.to_csv(root / "train_signals.csv", index=False)
    test.to_csv(root / "test_signals.csv", index=False)
    transactions(train_ids, train_dates, 0).to_parquet(root / "train_transactions.parquet")
    # Leave the last few test alerts with no transactions at all.
    transactions(test_ids, test_dates, silent_test_alerts).to_parquet(
        root / "test_transactions.parquet"
    )
    pd.DataFrame({"signal_id": test_ids, "ehtimollik": 0.5}).to_csv(
        root / "sample_submission.csv", index=False
    )
    return test_ids


@pytest.fixture
def fixture_data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    test_ids = _write_fixture(root)

    import src.pipeline as pipeline

    monkeypatch.setattr(config, "TRAIN_SIGNALS", root / "train_signals.csv")
    monkeypatch.setattr(config, "TEST_SIGNALS", root / "test_signals.csv")
    monkeypatch.setattr(config, "TRAIN_TX", root / "train_transactions.parquet")
    monkeypatch.setattr(config, "TEST_TX", root / "test_transactions.parquet")
    monkeypatch.setattr(config, "SAMPLE_SUBMISSION", root / "sample_submission.csv")
    # submission.py binds OUTPUTS at import time, so patching config alone
    # would write a fixture submission into the real outputs/ directory.
    import src.submission as submission

    monkeypatch.setattr(config, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(submission, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(pipeline, "SUMMARY_PATH", tmp_path / "summary.json")
    return root, test_ids, tmp_path


def _recipe():
    return {
        "families": ["direction_type", "base"],
        "weights": {"lightgbm": 0.5, "logreg": 0.5},
        "tuned": {"lightgbm": {"n_estimators": 30, "num_leaves": 8}, "logreg": {"C": 0.1}},
    }


def test_prediction_path_writes_a_valid_submission_for_every_test_id(fixture_data):
    import src.pipeline as pipeline

    root, test_ids, tmp_path = fixture_data
    monkeyed = _recipe()
    pipeline.MODELS = ("lightgbm", "logreg")

    from src.features import build_features, prepare_transactions
    from src.data import load_signals, load_transactions

    train_signals = load_signals(config.TRAIN_SIGNALS)
    train_tx = prepare_transactions(load_transactions(config.TRAIN_TX), train_signals)
    columns = list(
        build_features(train_tx, train_signals, families=monkeyed["families"]).columns
    )

    pipeline._predict_and_write(
        monkeyed["families"], columns, monkeyed["weights"], monkeyed["tuned"]
    )

    written = pd.read_csv(tmp_path / "outputs" / config.SUBMISSION_NAME)
    assert list(written.columns) == [config.ID, config.PROBA]
    assert list(written[config.ID]) == list(test_ids)
    assert written[config.PROBA].notna().all()
    assert written[config.PROBA].between(0, 1).all()
    assert written[config.PROBA].nunique() > 1


def test_alerts_with_no_transactions_still_receive_a_probability(fixture_data):
    import src.pipeline as pipeline

    root, test_ids, tmp_path = fixture_data
    recipe = _recipe()
    pipeline.MODELS = ("lightgbm", "logreg")

    from src.features import build_features, prepare_transactions
    from src.data import load_signals, load_transactions

    train_signals = load_signals(config.TRAIN_SIGNALS)
    train_tx = prepare_transactions(load_transactions(config.TRAIN_TX), train_signals)
    columns = list(
        build_features(train_tx, train_signals, families=recipe["families"]).columns
    )
    pipeline._predict_and_write(recipe["families"], columns, recipe["weights"], recipe["tuned"])

    written = pd.read_csv(tmp_path / "outputs" / config.SUBMISSION_NAME).set_index(config.ID)
    for silent in test_ids[-3:]:
        assert written.loc[silent, config.PROBA] == written.loc[silent, config.PROBA]


def test_a_corrupt_test_parquet_aborts_without_writing_anything(fixture_data):
    import src.pipeline as pipeline

    root, test_ids, tmp_path = fixture_data
    recipe = _recipe()
    pipeline.MODELS = ("lightgbm", "logreg")
    (root / "test_transactions.parquet").write_bytes(
        b"PAR1" + b"\xef\xbf\xbd" * 64 + b"\x10\x00\x00\x00PAR1"
    )

    with pytest.raises(CorruptParquetError):
        pipeline._predict_and_write(recipe["families"], ["base_cnt"], recipe["weights"], recipe["tuned"])

    assert not (tmp_path / "outputs" / config.SUBMISSION_NAME).exists()
