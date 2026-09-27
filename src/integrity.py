"""Checks that answer a single question: is our reported score real?

Two independent ways of fooling yourself, and the check that catches each.

1. A broken evaluation — leakage, misaligned indices, a target that reached the
   features. `permutation_test` destroys the target by shuffling it and reruns
   selection and scoring. An honest pipeline scores 0.5 on shuffled labels. Any
   meaningful lift means the number the real pipeline reports is not a
   measurement of anything.

2. Selecting on the data you then report. Column selection and Optuna both saw
   every training label, so the cross-validated figure is optimistic.
   `holdout_estimate` puts a slice away first, does all the choosing on the rest,
   and scores the untouched slice exactly once.
"""
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import SEED
from src.models import make_factory
from src.selection import rank_columns
from src.validation import evaluate


def judge_null(null_scores: Sequence[float], n_rows: int) -> tuple[str, str]:
    """Classify a null distribution: clean, optimistic, or leaking.

    Judge the MEAN, not the worst draw. With a handful of shuffles the largest
    draw sits one to two standard errors above the mean by chance, so a max-based
    threshold reports leakage on a perfectly clean pipeline.

    A null mean at 0.5 means the evaluation measures nothing when there is nothing
    to measure — the property we want. A mean a little above 0.5 is not leakage:
    it is the winner's curse of choosing the best of several candidates on the
    same data, and its size is how much that choosing inflates. A mean far above
    0.5 means the target reached the features.
    """
    mean = float(np.mean(null_scores))
    # Standard error of ROC-AUC, Hanley-McNeil style, adequate for a threshold.
    se = float(np.sqrt(1.0 / max(n_rows, 2)))
    excess = mean - 0.5

    if abs(excess) <= 2 * se:
        return "clean", f"null mean {mean:.5f} is within 2 SE ({2 * se:.5f}) of chance"
    if excess > 0.10:
        return "leaking", f"null mean {mean:.5f} is far above chance; the target reached the features"
    return (
        "optimistic",
        f"null mean {mean:.5f}: the selection step inflates by {excess:.4f} "
        f"when there is no signal to find",
    )


def _select_and_score(
    X: pd.DataFrame, y: pd.Series, model: str, sizes: Sequence[int]
) -> tuple[float, list[str]]:
    """Rank columns, pick the best-scoring prefix, return that score. One run of
    the same procedure the pipeline uses, so the null distribution is comparable."""
    ranked = rank_columns(X, y, model=model, n_repeats=3)
    best = (-np.inf, list(X.columns))
    for size in sizes:
        if size > X.shape[1]:
            continue
        columns = list(ranked.head(size).index)
        result = evaluate(make_factory(model), X[columns], y, n_repeats=2)
        if result.mean > best[0]:
            best = (result.mean, columns)
    return best[0], best[1]


def permutation_test(
    X: pd.DataFrame,
    y: pd.Series,
    model: str = "lightgbm",
    n_permutations: int = 5,
    sizes: Sequence[int] = (10, 20, 30),
    simulate_leak: bool = False,
) -> tuple[float, list[float]]:
    """Score the real labels, then score `n_permutations` shuffles of them.

    `simulate_leak` derives a feature from whichever labels are in use, which is
    what target leakage actually does — the leaked column follows the shuffle.
    The tests use it to prove this check can detect leakage rather than always
    reporting chance.
    """
    def frame_for(labels: pd.Series) -> pd.DataFrame:
        if not simulate_leak:
            return X
        return X.assign(_leak=labels.to_numpy().astype(float))

    observed, _ = _select_and_score(frame_for(y), y, model, sizes)

    null_scores: list[float] = []
    rng = np.random.default_rng(SEED)
    for _ in range(n_permutations):
        shuffled = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        score, _ = _select_and_score(frame_for(shuffled), shuffled, model, sizes)
        null_scores.append(score)
    return observed, null_scores


def holdout_estimate(
    X: pd.DataFrame,
    y: pd.Series,
    model: str = "lightgbm",
    sizes: Sequence[int] = (10, 15, 20, 25, 30, 40),
    holdout_fraction: float = 0.2,
    tune_trials: int = 0,
) -> dict:
    """Select and tune on one slice, score the untouched slice once."""
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=holdout_fraction, random_state=SEED, stratify=y
    )

    ranked = rank_columns(X_train, y_train, model=model)
    best_mean, best_std, columns = -np.inf, None, list(X_train.columns)
    for size in sizes:
        if size > X_train.shape[1]:
            continue
        candidate = list(ranked.head(size).index)
        result = evaluate(make_factory(model), X_train[candidate], y_train)
        if result.mean > best_mean:
            best_mean, best_std, columns = result.mean, result.std, candidate

    params = None
    if tune_trials:
        from src.tuning import tune

        params = tune(model, X_train[columns], y_train, n_trials=tune_trials)

    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_frame = pd.DataFrame(
        imputer.fit_transform(X_train[columns]), columns=columns, index=X_train.index
    )
    hold_frame = pd.DataFrame(
        imputer.transform(X_hold[columns]), columns=columns, index=X_hold.index
    )
    estimator = make_factory(model, params)(SEED)
    estimator.fit(fit_frame, y_train)
    holdout_auc = float(roc_auc_score(y_hold, estimator.predict_proba(hold_frame)[:, 1]))

    return {
        "cv_mean": float(best_mean),
        "cv_std": float(best_std) if best_std is not None else None,
        "holdout_auc": holdout_auc,
        "optimism": float(best_mean) - holdout_auc,
        "n_train": len(y_train),
        "n_holdout": len(y_hold),
        "columns": columns,
        "params": params,
    }
