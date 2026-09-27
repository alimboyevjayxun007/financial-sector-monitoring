"""Checks that answer 'is our reported number real, or are we fooling ourselves?'"""
import numpy as np
import pandas as pd

from src.integrity import holdout_estimate, permutation_test


def _learnable(n=500, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    X = pd.DataFrame({"good": signal + rng.normal(scale=0.5, size=n)}, index=range(n))
    for i in range(4):
        X[f"noise{i}"] = rng.normal(size=n)
    y = pd.Series((signal > 0).astype(int), index=range(n))
    return X, y


def test_permutation_test_returns_chance_on_shuffled_labels():
    """The whole point: with the target destroyed, an honest pipeline scores 0.5.
    Anything meaningfully above chance means the evaluation itself leaks."""
    X, y = _learnable()
    observed, null_scores = permutation_test(X, y, model="logreg", n_permutations=3, sizes=(2, 4))
    assert observed > 0.6
    assert len(null_scores) == 3
    assert max(null_scores) < 0.60, null_scores


def test_permutation_test_detects_a_leaking_evaluation():
    """When the target reaches the features, the leaked column follows the
    shuffle and the null run scores high. This proves the check can actually
    catch leakage rather than always reporting chance."""
    X, y = _learnable()
    observed, null_scores = permutation_test(
        X, y, model="logreg", n_permutations=2, sizes=(2,), simulate_leak=True
    )
    assert max(null_scores) > 0.9, null_scores


def test_holdout_estimate_reports_both_numbers_on_disjoint_rows():
    X, y = _learnable(n=800)
    report = holdout_estimate(X, y, model="logreg", sizes=(2, 4), tune_trials=0)
    assert {"cv_mean", "cv_std", "holdout_auc", "n_train", "n_holdout", "columns"} <= set(report)
    assert report["n_train"] + report["n_holdout"] == len(y)
    assert report["holdout_auc"] > 0.6


def test_null_verdict_uses_the_mean_not_the_worst_draw():
    """With a handful of shuffles the largest draw sits well above the mean by
    chance; judging leakage on the max cries wolf. The mean is the statistic."""
    from src.integrity import judge_null

    clean = [0.492, 0.516, 0.483, 0.498, 0.491]   # mean 0.496 — chance
    leaking = [0.94, 0.95, 0.93, 0.96, 0.94]

    verdict, detail = judge_null(clean, n_rows=14000)
    assert verdict == "clean", detail

    verdict, detail = judge_null(leaking, n_rows=14000)
    assert verdict == "leaking", detail


def test_null_verdict_flags_a_selection_procedure_that_inflates():
    """A null mean a couple of standard errors above 0.5 is not leakage but is
    not nothing either: it measures how much the selection step inflates."""
    from src.integrity import judge_null

    inflated = [0.5198, 0.5319, 0.5135, 0.5195, 0.5071]  # mean 0.518
    verdict, detail = judge_null(inflated, n_rows=14000)
    assert verdict == "optimistic", detail
    assert "0.018" in detail or "0.0184" in detail
