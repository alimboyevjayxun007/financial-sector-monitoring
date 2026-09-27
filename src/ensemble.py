import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def rank_average(
    predictions: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Weighted average of per-model rank percentiles. Scale invariant."""
    names = list(predictions)
    if weights is None:
        weights = {name: 1.0 / len(names) for name in names}
    total = sum(weights[name] for name in names)
    n = len(next(iter(predictions.values())))
    blended = np.zeros(n)
    for name in names:
        percentile = (rankdata(predictions[name]) - 1) / max(n - 1, 1)
        blended += (weights[name] / total) * percentile
    return blended


def fit_weights(oof: dict[str, np.ndarray], y: pd.Series) -> dict[str, float]:
    """Simplex-constrained weights maximizing OOF ROC-AUC."""
    names = list(oof)
    start = np.full(len(names), 1.0 / len(names))

    def negative_auc(raw: np.ndarray) -> float:
        clipped = np.clip(raw, 0.0, None)
        if clipped.sum() == 0:
            return 0.0
        candidate = dict(zip(names, clipped))
        return -roc_auc_score(y, rank_average(oof, candidate))

    result = minimize(
        negative_auc,
        start,
        method="Nelder-Mead",
        options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-6},
    )
    raw = np.clip(result.x, 0.0, None)
    if raw.sum() == 0:
        raw = start
    raw = raw / raw.sum()
    return {name: float(value) for name, value in zip(names, raw)}


def score_blend(
    oof_per_repeat: dict[str, np.ndarray],
    y: pd.Series,
    weights: dict[str, float],
) -> tuple[float, float, list[float]]:
    """Score a blend the way a single model is scored: per repeat, then averaged.

    Scoring on repeat-averaged OOF instead inflates AUC on its own, which would
    make the ensemble look better than the single models it is compared against.
    """
    n_repeats = next(iter(oof_per_repeat.values())).shape[0]
    scores = [
        float(
            roc_auc_score(
                y,
                rank_average(
                    {name: vectors[repeat] for name, vectors in oof_per_repeat.items()},
                    weights,
                ),
            )
        )
        for repeat in range(n_repeats)
    ]
    return float(np.mean(scores)), float(np.std(scores)), scores


def choose(
    oof: dict[str, np.ndarray],
    y: pd.Series,
    min_gain: float = 0.001,
) -> tuple[dict[str, float], str]:
    """Return fitted weights only when they beat equal weights by min_gain."""
    equal = {name: 1.0 / len(oof) for name in oof}
    equal_auc = roc_auc_score(y, rank_average(oof, equal))
    fitted = fit_weights(oof, y)
    fitted_auc = roc_auc_score(y, rank_average(oof, fitted))
    if fitted_auc - equal_auc >= min_gain:
        return fitted, "fitted"
    return equal, "equal"
