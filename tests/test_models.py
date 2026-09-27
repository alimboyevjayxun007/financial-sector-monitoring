import numpy as np
import pandas as pd
import pytest

from src.models import MODEL_FACTORIES, bagged_predict, default_params, make_factory


def _dataset(n=300):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    return X, y


@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_every_factory_builds_a_fittable_classifier(name):
    X, y = _dataset()
    model = make_factory(name)(7)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(y),)
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_defaults_are_regularization_biased(name):
    params = default_params(name)
    assert isinstance(params, dict)


def test_lightgbm_defaults_constrain_capacity():
    params = default_params("lightgbm")
    assert params["num_leaves"] <= 31
    assert params["min_child_samples"] >= 40
    assert params["reg_lambda"] >= 1.0
    assert params["learning_rate"] <= 0.05


def test_bagged_predict_averages_over_seeds_and_is_deterministic():
    X, y = _dataset()
    first = bagged_predict("lightgbm", default_params("lightgbm"), X, y, X, seeds=[1, 2, 3])
    second = bagged_predict("lightgbm", default_params("lightgbm"), X, y, X, seeds=[1, 2, 3])
    assert first.shape == (len(y),)
    np.testing.assert_allclose(first, second)


def test_logreg_handles_an_all_nan_column():
    X, y = _dataset()
    X["dead"] = np.nan
    model = make_factory("logreg")(7)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert np.isfinite(proba).all()
