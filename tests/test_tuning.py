import numpy as np
import pandas as pd

from src.models import make_factory
from src.tuning import SEARCH_SPACES, tune


def _dataset(n=300):
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + rng.normal(scale=0.6, size=n) > 0).astype(int))
    return X, y


def test_every_model_has_a_search_space():
    assert set(SEARCH_SPACES) == {"lightgbm", "xgboost", "catboost", "logreg"}


def test_tune_returns_params_that_build_a_working_model():
    X, y = _dataset()
    params = tune("lightgbm", X, y, n_trials=3)
    assert isinstance(params, dict)
    model = make_factory("lightgbm", params)(1)
    model.fit(X, y)
    assert model.predict_proba(X).shape == (len(y), 2)


def test_tune_is_deterministic_for_a_fixed_trial_count():
    X, y = _dataset()
    first = tune("lightgbm", X, y, n_trials=3)
    second = tune("lightgbm", X, y, n_trials=3)
    assert first == second


def test_tuned_lightgbm_params_keep_subsample_effective():
    """LightGBM ignores subsample unless subsample_freq > 0, and Optuna's
    best_params drops constants that were never suggested."""
    X, y = _dataset()
    params = tune("lightgbm", X, y, n_trials=3)
    assert params["subsample_freq"] >= 1
