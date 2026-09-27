import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.validation import CVResult, evaluate, log_run


def _dataset(n=400):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    return X, y


def test_evaluate_returns_score_of_mean_minus_std():
    X, y = _dataset()
    result = evaluate(lambda seed: LogisticRegression(random_state=seed), X, y, n_repeats=2)
    assert isinstance(result, CVResult)
    assert result.score == result.mean - result.std
    assert 0.5 < result.mean <= 1.0


def test_evaluate_produces_one_oof_prediction_per_row():
    X, y = _dataset()
    result = evaluate(lambda seed: LogisticRegression(random_state=seed), X, y, n_repeats=2)
    assert result.oof.shape == (len(y),)
    assert np.isfinite(result.oof).all()


def test_evaluate_is_deterministic():
    X, y = _dataset()
    first = evaluate(lambda seed: LogisticRegression(random_state=seed), X, y, n_repeats=2)
    second = evaluate(lambda seed: LogisticRegression(random_state=seed), X, y, n_repeats=2)
    assert first.mean == second.mean
    np.testing.assert_allclose(first.oof, second.oof)


def test_evaluate_tolerates_an_all_nan_column():
    X, y = _dataset()
    X["dead"] = np.nan
    result = evaluate(lambda seed: LogisticRegression(random_state=seed), X, y, n_repeats=2)
    assert np.isfinite(result.oof).all()


def test_log_run_appends_a_row(tmp_path, monkeypatch):
    import src.validation as validation

    monkeypatch.setattr(validation, "LOG_PATH", tmp_path / "log.csv")
    X, y = _dataset()
    result = evaluate(lambda seed: LogisticRegression(random_state=seed), X, y, n_repeats=2)
    log_run(result, families=["base"], n_columns=2, model="logreg", params={"C": 1.0})
    log_run(result, families=["base"], n_columns=2, model="logreg", params={"C": 1.0})
    rows = pd.read_csv(tmp_path / "log.csv")
    assert len(rows) == 2
    assert {"timestamp", "families", "n_columns", "model", "cv_mean", "cv_std", "score"} <= set(rows.columns)


def test_cvresult_exposes_per_repeat_oof_so_blends_can_be_scored_the_same_way():
    """Scoring a blend on repeat-averaged OOF inflates AUC relative to the
    per-repeat mean the single-model numbers use. Comparing the two is invalid,
    so per-repeat vectors must be available to score a blend consistently."""
    X, y = _dataset()
    result = evaluate(lambda seed: LogisticRegression(random_state=seed), X, y, n_repeats=3)
    assert result.oof_per_repeat.shape == (3, len(y))
    np.testing.assert_allclose(result.oof_per_repeat.mean(axis=0), result.oof)
    assert len(result.repeat_scores) == 3
    assert result.mean == float(np.mean(result.repeat_scores))
