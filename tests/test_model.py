import numpy as np
import pandas as pd

from src.model import cross_validate, train


def _synthetic_classification_data(n=400, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    # target correlated with a linear combination + noise, imbalanced like the real data
    score = X["f0"] * 2 - X["f1"] + rng.normal(scale=0.5, size=n)
    threshold = np.quantile(score, 0.83)  # ~17% positive, matches real target rate
    y = pd.Series((score > threshold).astype(int))
    return X, y


def test_train_returns_fitted_model_with_predict_proba():
    X, y = _synthetic_classification_data()
    model = train(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape[0] == len(X)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_cross_validate_returns_reasonable_auc_on_separable_data():
    X, y = _synthetic_classification_data(n=600)
    auc = cross_validate(X, y, n_splits=3)
    assert 0.0 <= auc <= 1.0
    # the synthetic target is a real (if noisy) function of the features,
    # so a working CV loop should clear a low bar comfortably
    assert auc > 0.6


def test_cross_validate_is_near_chance_on_pure_noise():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(500, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.choice([0, 1], size=500, p=[0.83, 0.17]))
    auc = cross_validate(X, y, n_splits=3)
    # unrelated target -> should not be able to fake a strong AUC
    assert 0.3 < auc < 0.7
