import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.pipeline import predict_from_saved


def test_predict_from_saved_refuses_when_no_run_has_been_recorded(tmp_path, monkeypatch):
    import src.pipeline as pipeline

    monkeypatch.setattr(pipeline, "SUMMARY_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError) as excinfo:
        predict_from_saved()
    assert "src.pipeline" in str(excinfo.value)


def test_predict_from_saved_reads_the_recorded_families_and_params(tmp_path, monkeypatch):
    import src.pipeline as pipeline

    recorded = {
        "families": ["base"],
        "n_columns": 2,
        "weights": {"lightgbm": 1.0},
        "tuned_params": {"lightgbm": {"n_estimators": 10}},
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(recorded))
    monkeypatch.setattr(pipeline, "SUMMARY_PATH", path)

    captured = {}

    def fake_predict(families, columns, weights, tuned):
        captured.update(families=families, columns=columns, weights=weights, tuned=tuned)
        return {"adversarial_auc": 0.5}

    monkeypatch.setattr(pipeline, "_predict_and_write", fake_predict)
    monkeypatch.setattr(pipeline, "load_selection", lambda: {"columns": ["base_cnt", "base_sum"]})

    assert predict_from_saved() == {"adversarial_auc": 0.5}
    assert captured["families"] == ["base"]
    assert captured["columns"] == ["base_cnt", "base_sum"]
    assert captured["weights"] == {"lightgbm": 1.0}
    assert captured["tuned"] == {"lightgbm": {"n_estimators": 10}}


def test_predict_from_saved_rejects_a_selection_file_that_disagrees_with_the_summary(
    tmp_path, monkeypatch
):
    """run() writes selected_features.json before summary.json, so a crash during
    tuning leaves a new selection beside a stale summary."""
    import src.pipeline as pipeline

    path = tmp_path / "summary.json"
    path.write_text(json.dumps({
        "families": ["base"],
        "n_columns": 2,
        "weights": {"lightgbm": 1.0},
        "tuned_params": {"lightgbm": {}},
    }))
    monkeypatch.setattr(pipeline, "SUMMARY_PATH", path)
    monkeypatch.setattr(pipeline, "load_selection", lambda: {"columns": ["a", "b", "c"]})

    with pytest.raises(ValueError) as excinfo:
        predict_from_saved()
    assert "stale" in str(excinfo.value).lower() or "disagree" in str(excinfo.value).lower()


def test_predict_from_saved_records_the_drift_result_into_the_summary(tmp_path, monkeypatch):
    """The adversarial AUC is a headline finding; it must survive into the
    record the site and notebook read, not just stdout."""
    import src.pipeline as pipeline

    path = tmp_path / "summary.json"
    path.write_text(json.dumps({
        "families": ["base"],
        "n_columns": 2,
        "weights": {"lightgbm": 1.0},
        "tuned_params": {"lightgbm": {}},
    }))
    monkeypatch.setattr(pipeline, "SUMMARY_PATH", path)
    monkeypatch.setattr(pipeline, "load_selection", lambda: {"columns": ["a", "b"]})
    monkeypatch.setattr(
        pipeline, "_predict_and_write",
        lambda *a, **k: {"adversarial_auc": 0.4921, "top_drifting": {"base_span": 4203.0}},
    )

    predict_from_saved()
    recorded = json.loads(path.read_text())
    assert recorded["adversarial_auc"] == 0.4921
    assert recorded["top_drifting"] == {"base_span": 4203.0}
    assert recorded["families"] == ["base"]
