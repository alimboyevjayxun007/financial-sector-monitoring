import logging
from collections.abc import Callable

import optuna
import pandas as pd

from src.config import SEED
from src.models import make_factory
from src.validation import evaluate, log_run

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)


def _lightgbm_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 31),
        "min_child_samples": trial.suggest_int("min_child_samples", 40, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
    }


def _xgboost_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
    }


def _catboost_space(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "depth": trial.suggest_int("depth", 2, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.4, 1.0),
    }


def _logreg_space(trial: optuna.Trial) -> dict:
    return {"C": trial.suggest_float("C", 0.001, 10.0, log=True)}


# Values fixed rather than searched. Optuna's best_params reports only
# suggested values, so these must be merged back in or they are silently lost
# (LightGBM ignores subsample entirely when subsample_freq is 0).
CONSTANTS: dict[str, dict] = {
    "lightgbm": {"subsample_freq": 1},
    "xgboost": {},
    "catboost": {},
    "logreg": {},
}

SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "lightgbm": _lightgbm_space,
    "xgboost": _xgboost_space,
    "catboost": _catboost_space,
    "logreg": _logreg_space,
}


def tune(name: str, X: pd.DataFrame, y: pd.Series, n_trials: int = 40) -> dict:
    """Optuna search judged on mean - std, never on a single split."""
    space = SEARCH_SPACES[name]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        result = evaluate(make_factory(name, params), X, y, n_repeats=2)
        log_run(
            result,
            families=["tuning"],
            n_columns=X.shape[1],
            model=name,
            params=params,
        )
        return result.score

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**CONSTANTS[name], **study.best_params}
