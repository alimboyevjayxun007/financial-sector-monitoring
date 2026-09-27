from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import N_JOBS, SEED

DEFAULTS: dict[str, dict] = {
    "lightgbm": {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 16,
        "min_child_samples": 60,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_lambda": 5.0,
        "reg_alpha": 1.0,
    },
    "xgboost": {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 5.0,
        "reg_alpha": 1.0,
    },
    "catboost": {
        "iterations": 700,
        "learning_rate": 0.03,
        "depth": 4,
        "l2_leaf_reg": 8.0,
        "rsm": 0.7,
    },
    "logreg": {"C": 0.1},
}


def default_params(name: str) -> dict:
    return dict(DEFAULTS[name])


def _lightgbm(params: dict) -> Callable[[int], object]:
    return lambda seed: LGBMClassifier(random_state=seed, verbose=-1, n_jobs=N_JOBS, **params)


def _xgboost(params: dict) -> Callable[[int], object]:
    return lambda seed: XGBClassifier(
        random_state=seed, n_jobs=N_JOBS, eval_metric="auc", tree_method="hist", **params
    )


def _catboost(params: dict) -> Callable[[int], object]:
    return lambda seed: CatBoostClassifier(
        random_seed=seed, verbose=0, allow_writing_files=False, thread_count=N_JOBS, **params
    )


def _logreg(params: dict) -> Callable[[int], object]:
    return lambda seed: make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed, **params),
    )


MODEL_FACTORIES: dict[str, Callable[[dict], Callable[[int], object]]] = {
    "lightgbm": _lightgbm,
    "xgboost": _xgboost,
    "catboost": _catboost,
    "logreg": _logreg,
}


def make_factory(name: str, params: dict | None = None) -> Callable[[int], object]:
    return MODEL_FACTORIES[name](default_params(name) if params is None else dict(params))


def bagged_predict(
    name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    seeds: Sequence[int] | None = None,
) -> np.ndarray:
    """Fit one model per seed on the full training set and average test probabilities."""
    seeds = list(range(SEED, SEED + 5)) if seeds is None else list(seeds)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fitted_train = pd.DataFrame(
        imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    fitted_test = pd.DataFrame(
        imputer.transform(X_test), index=X_test.index, columns=X_test.columns
    )
    factory = MODEL_FACTORIES[name](dict(params))
    total = np.zeros(len(X_test))
    for seed in seeds:
        model = factory(seed)
        model.fit(fitted_train, y_train)
        total += model.predict_proba(fitted_test)[:, 1]
    return total / len(seeds)
