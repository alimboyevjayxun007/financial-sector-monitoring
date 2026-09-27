"""End-to-end run: features, selection, tuning, ensemble, submission."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.adversarial import drift_report
from src.data import CorruptParquetError, load_signals, load_transactions
from src.ensemble import choose, rank_average, score_blend
from src.features import FEATURE_FAMILIES, build_features, prepare_transactions
from src.models import bagged_predict, make_factory
from src.selection import (
    eliminate_columns,
    load_selection,
    save_selection,
    select_columns_globally,
    select_families,
)
from src.submission import write
from src.tuning import tune
from src.validation import evaluate, log_run

MODELS = ("lightgbm", "xgboost", "catboost", "logreg")

SUMMARY_PATH = config.EXPERIMENTS / "summary.json"


def _family_frames(tx: pd.DataFrame, signals: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        name: build_features(tx, signals, families=[name]) for name in FEATURE_FAMILIES
    }


def run(tune_trials: int = 40, skip_test: bool = False) -> dict:
    train_signals = load_signals(config.TRAIN_SIGNALS)
    y = train_signals[config.TARGET]
    train_tx = prepare_transactions(
        load_transactions(config.TRAIN_TX), train_signals
    )

    print("Ranking feature families (recorded for the write-up)...")
    by_family = _family_frames(train_tx, train_signals)
    families, history = select_families(by_family, y)
    print("  family-level search would keep:", families)

    # Selection happens across ALL columns, not within the chosen families:
    # family-level greedy search discards every column of a rejected family,
    # and several columns in the global top 15 come from rejected families.
    X_all = pd.concat(by_family.values(), axis=1)
    print(f"Selecting columns globally from {X_all.shape[1]}...")
    columns, size_trace = select_columns_globally(X_all, y)
    X = X_all[columns]
    all_families = list(by_family)
    save_selection(columns, all_families)
    for row in size_trace:
        print(f"  top{row['size']:<3} mean={row['mean']:.5f} std={row['std']:.5f} "
              f"score={row['score']:.5f}")
    print(f"  kept {len(columns)} columns")

    print("Tuning...")
    tuned, oof, per_repeat, scores, means = {}, {}, {}, {}, {}
    for name in MODELS:
        params = tune(name, X, y, n_trials=tune_trials)
        result = evaluate(make_factory(name, params), X, y)
        log_run(result, families=all_families, n_columns=X.shape[1], model=name, params=params)
        tuned[name], oof[name] = params, result.oof
        per_repeat[name] = result.oof_per_repeat
        scores[name], means[name] = result.score, result.mean
        print(f"  {name}: mean={result.mean:.5f} std={result.std:.5f} score={result.score:.5f}")

    weights, method = choose(oof, y)
    # Scored per repeat then averaged, exactly as each single model above is.
    # Scoring on repeat-averaged OOF instead inflates AUC by roughly 0.004 here
    # and would make the ensemble look better than the models it is compared to.
    blend_mean, blend_std, _ = score_blend(per_repeat, y, weights)
    best_single = max(means, key=means.get)
    print(
        f"Ensemble ({method}): mean={blend_mean:.5f} std={blend_std:.5f} "
        f"score={blend_mean - blend_std:.5f}"
    )
    print(
        f"  best single model: {best_single} mean={means[best_single]:.5f} "
        f"-> ensemble gain {blend_mean - means[best_single]:+.5f}"
    )

    summary = {
        "families": all_families,
        "family_level_would_keep": families,
        "column_size_trace": size_trace,
        "n_columns": len(columns),
        "model_scores": scores,
        "weights": weights,
        "weighting": method,
        "ensemble_cv_mean": blend_mean,
        "ensemble_cv_std": blend_std,
        "ensemble_score": blend_mean - blend_std,
        "best_single_model": best_single,
        "best_single_cv_mean": means[best_single],
        "ensemble_gain_over_best_single": blend_mean - means[best_single],
        "model_cv_means": means,
        "selected_columns": columns,
        "selection_history": history,
        "tuned_params": tuned,
    }
    config.EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=float))

    if skip_test:
        print("skip_test=True — stopping before prediction.")
        return summary

    drift = _predict_and_write(all_families, columns, weights, tuned, X=X, y=y)
    summary.update(drift)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=float))
    return summary


DRIFT_WARNING_AUC = 0.75


def _predict_and_write(
    families: list[str],
    columns: list[str],
    weights: dict[str, float],
    tuned: dict[str, dict],
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> dict:
    """Build test features, report drift, and write the validated submission."""
    train_signals = load_signals(config.TRAIN_SIGNALS)
    if X is None or y is None:
        y = train_signals[config.TARGET]
        train_tx = prepare_transactions(load_transactions(config.TRAIN_TX), train_signals)
        X = build_features(train_tx, train_signals, families=families)[columns]

    try:
        test_signals = load_signals(config.TEST_SIGNALS)
        test_tx = prepare_transactions(load_transactions(config.TEST_TX), test_signals)
    except CorruptParquetError as exc:
        print(f"\nCannot predict: {exc}")
        raise

    X_test = build_features(
        test_tx, test_signals, families=families, reference=train_signals
    )[columns]

    auc, drifting = drift_report(X, X_test)
    print(f"Adversarial validation AUC: {auc:.4f}")
    print("Top drifting columns:\n", drifting.head(10))
    if auc > DRIFT_WARNING_AUC:
        print(
            f"\nWARNING: train and test are separable at AUC {auc:.4f} "
            f"(> {DRIFT_WARNING_AUC}). The features above differ systematically "
            f"between splits; the CV estimate may not transfer. Inspect them "
            f"before trusting this submission."
        )

    predictions = {
        name: bagged_predict(name, tuned[name], X, y, X_test) for name in MODELS
    }
    blended = rank_average(predictions, weights)

    frame = pd.DataFrame({config.ID: X_test.index, config.PROBA: blended})
    expected = pd.read_csv(config.SAMPLE_SUBMISSION)[config.ID]
    path = write(frame, expected)
    print(f"Wrote {path}")

    return {"adversarial_auc": auc, "top_drifting": drifting.head(10).to_dict()}


def predict_from_saved():
    """Reproduce the submission from a recorded run, without re-searching."""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found. Run `python -m src.pipeline` first; "
            f"it records the chosen families, the frozen columns, and the tuned params."
        )
    summary = json.loads(SUMMARY_PATH.read_text())
    columns = load_selection()["columns"]
    # run() writes the selection file before summary.json, so a crash during
    # tuning leaves a new selection beside a stale summary. Predicting from the
    # pair would use one run's columns with another run's hyperparameters.
    if len(columns) != summary["n_columns"]:
        raise ValueError(
            f"selected_features.json and summary.json disagree: "
            f"{len(columns)} columns vs {summary['n_columns']} recorded. "
            f"The artifacts are stale; re-run `python -m src.pipeline`."
        )
    drift = _predict_and_write(
        summary["families"], columns, summary["weights"], summary["tuned_params"]
    )
    # The adversarial AUC is a headline finding; keep it in the record the site
    # and notebook read, not only on stdout.
    summary.update(drift)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=float))
    return drift


if __name__ == "__main__":
    run()
