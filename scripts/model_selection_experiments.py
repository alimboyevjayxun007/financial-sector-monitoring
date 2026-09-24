"""Model selection experiments — the reproducible source of every number
cited in README.md sections 5.1/5.1.1/5.2.

This script exists because an earlier version of the project cited detailed
CV numbers (mean AUC, std, p-values) that were reported by a background
subagent but never actually committed as runnable code — an audit correctly
flagged that as unverifiable and indistinguishable from confabulation. This
script is the fix: every number in the README's model-comparison tables must
trace back to a run of this file.

Run: python3 scripts/model_selection_experiments.py
Output: prints a full report AND writes scripts/model_selection_results.csv

Runtime: ~3-6 minutes (SVC and the LightGBM search are the slow parts).

Statistical note: naive paired t-tests on RepeatedStratifiedKFold scores are
anti-conservative (Dietterich 1998; Nadeau & Bengio 2003) because k-fold
resampling produces overlapping train/validation sets across folds/repeats,
violating the independence assumption. This script reports BOTH the naive
paired t-test and the Nadeau-Bengio corrected version, and the corrected
p-value is what README actually cites. We also ran ~10 comparisons here, so
even the corrected p-value should be read as one data point, not proof —
README says so explicitly rather than overclaiming significance.
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    ParameterSampler,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

RESULTS_PATH = pathlib.Path(__file__).resolve().parent / "model_selection_results.csv"

BASE_18 = [
    c
    for c in config.FEATURE_COLUMNS
    if c not in ("hour_entropy", "hour_maxshare", "dow_entropy", "dow_maxshare")
]
FULL_22 = config.FEATURE_COLUMNS
assert len(BASE_18) == 18 and len(FULL_22) == 22, (len(BASE_18), len(FULL_22))


def lr_pipeline(**kwargs):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", **kwargs)),
        ]
    )


def cv_scores(X, y, make_model, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        m = make_model()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = m.predict_proba(X.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va_idx], p))
    return np.array(aucs)


def repeated_cv_scores(X, y, make_model, n_splits=5, n_repeats=5, random_state=42):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    aucs = []
    for tr_idx, va_idx in rskf.split(X, y):
        m = make_model()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = m.predict_proba(X.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va_idx], p))
    return np.array(aucs)


def naive_paired_ttest(diffs):
    t, p = stats.ttest_1samp(diffs, 0.0)
    return t, p


def nadeau_bengio_test(diffs, k=5):
    """Corrected variance paired t-test for k-fold (repeated) CV comparisons.
    Ref: Nadeau & Bengio (2003), 'Inference for the Generalization Error'."""
    n = len(diffs)
    mean_diff = diffs.mean()
    var_diff = diffs.var(ddof=1)
    n_test_over_train = 1.0 / (k - 1)
    correction = 1.0 / n + n_test_over_train
    se = np.sqrt(correction * var_diff) if var_diff > 0 else 1e-12
    t = mean_diff / se
    df = n - 1
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return t, p


def main():
    t0 = time.time()
    train = pd.read_parquet(config.TRAIN_FEATURES_PATH)
    test = pd.read_parquet(config.TEST_FEATURES_PATH)
    y = train[config.TARGET_COL]
    X18 = train[BASE_18]
    X22 = train[FULL_22]

    results = []

    def record(name, feat_set, aucs, note=""):
        results.append(
            {
                "model": name,
                "features": feat_set,
                "n_folds": len(aucs),
                "auc_mean": aucs.mean(),
                "auc_std": aucs.std(),
                "note": note,
            }
        )
        print(f"[{name} | {feat_set}] AUC = {aucs.mean():.4f} +/- {aucs.std():.4f}  ({note})")

    print("=" * 70)
    print("PART 1: model family comparison on the base 18-feature set (5-fold CV)")
    print("=" * 70)

    record("HistGradientBoosting(depth=6)", "18", cv_scores(X18, y, lambda: HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=6, class_weight="balanced",
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42)))

    record("HistGradientBoosting(depth=3)", "18", cv_scores(X18, y, lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=3, class_weight="balanced",
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42)))

    record("RandomForest", "18", cv_scores(X18, y, lambda: RandomForestClassifier(
        n_estimators=400, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)))

    record("SVC(RBF)", "18", cv_scores(X18, y, lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
    ])))

    record("MLP(16,alpha=1.0)", "18", cv_scores(X18, y, lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=1000, random_state=42)),
    ])))

    lr18_scores = cv_scores(X18, y, lr_pipeline)
    record("LogisticRegression", "18", lr18_scores, note="baseline, pre-entropy")

    def lr_poly():
        return Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])

    record("LogisticRegression+poly2", "18", cv_scores(X18, y, lr_poly))

    print()
    print("=" * 70)
    print("PART 2: does adding hour/dow entropy (18 -> 22 features) help?")
    print("=" * 70)

    lr22_scores = cv_scores(X22, y, lr_pipeline)
    record("LogisticRegression", "22", lr22_scores, note="with hour/dow entropy")

    lr18_rep = repeated_cv_scores(X18, y, lr_pipeline)
    lr22_rep = repeated_cv_scores(X22, y, lr_pipeline)
    record("LogisticRegression (5x5 repeat)", "18", lr18_rep)
    record("LogisticRegression (5x5 repeat)", "22", lr22_rep)

    diffs = lr22_rep - lr18_rep
    t_naive, p_naive = naive_paired_ttest(diffs)
    t_nb, p_nb = nadeau_bengio_test(diffs, k=5)
    print(f"\n18 -> 22 feature diff (5x5 repeat): mean={diffs.mean():.4f}")
    print(f"  naive paired t-test:            t={t_naive:.3f}, p={p_naive:.4f}  <- anti-conservative, DO NOT cite alone")
    print(f"  Nadeau-Bengio corrected t-test:  t={t_nb:.3f}, p={p_nb:.4f}  <- this is what README cites")

    print()
    print("=" * 70)
    print("PART 3: tuned LightGBM (randomized search) on 22 features")
    print("=" * 70)

    if _HAS_LGBM:
        param_dist = {
            "num_leaves": [7, 15, 31, 63],
            "max_depth": [2, 3, 4, -1],
            "min_child_samples": [20, 50, 100, 200],
            "reg_alpha": [0.0, 0.1, 1.0, 5.0],
            "reg_lambda": [0.0, 0.1, 1.0, 5.0],
            "feature_fraction": [0.6, 0.8, 1.0],
            "bagging_fraction": [0.6, 0.8, 1.0],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
        }
        sampler = list(ParameterSampler(param_dist, n_iter=15, random_state=42))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_mean, best_params = -1, None
        for params in sampler:
            aucs = []
            for tr_idx, va_idx in skf.split(X22, y):
                Xtr, Xva = X22.iloc[tr_idx], X22.iloc[va_idx]
                ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
                m = LGBMClassifier(
                    n_estimators=500, class_weight="balanced", random_state=42,
                    verbosity=-1, **params,
                )
                m.fit(
                    Xtr, ytr, eval_set=[(Xva, yva)],
                    callbacks=[__import__("lightgbm").early_stopping(20, verbose=False)],
                )
                p = m.predict_proba(Xva)[:, 1]
                aucs.append(roc_auc_score(yva, p))
            mean_auc = np.mean(aucs)
            if mean_auc > best_mean:
                best_mean, best_params = mean_auc, params
        record("LightGBM (best of 15-config search)", "22", np.array([best_mean]),
               note=f"best_params={best_params}")
    else:
        print("lightgbm not installed, skipping")

    print()
    print("=" * 70)
    print("PART 4: probability calibration")
    print("=" * 70)

    def calibrated_lr():
        base = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        return CalibratedClassifierCV(base, method="sigmoid", cv=5)

    cal_scores = cv_scores(X22, y, calibrated_lr)
    record("CalibratedClassifierCV(LR, sigmoid)", "22", cal_scores,
           note="AUC should ~match uncalibrated LR; calibration changes probabilities, not ranking")

    # Actual calibration quality: mean predicted vs true base rate, uncalibrated vs calibrated
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    uncal_preds, cal_preds, true_vals = [], [], []
    for tr_idx, va_idx in skf.split(X22, y):
        Xtr, Xva = X22.iloc[tr_idx], X22.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        uncal_preds.append(m.predict_proba(sc.transform(Xva))[:, 1])
        mc = calibrated_lr().fit(Xtr, ytr)
        cal_preds.append(mc.predict_proba(Xva)[:, 1])
        true_vals.append(yva.values)
    uncal_preds = np.concatenate(uncal_preds)
    cal_preds = np.concatenate(cal_preds)
    true_vals = np.concatenate(true_vals)
    print(f"true base rate:              {true_vals.mean():.4f}")
    print(f"uncalibrated mean pred:      {uncal_preds.mean():.4f}  (class_weight='balanced' pushes this toward 0.5)")
    print(f"calibrated mean pred:        {cal_preds.mean():.4f}  (should track true base rate much more closely)")

    print()
    print("=" * 70)
    print("PART 5: adversarial validation (train vs test, 22-feature set)")
    print("=" * 70)

    Xadv = pd.concat([train[FULL_22], test[FULL_22]], ignore_index=True)
    yadv = np.array([0] * len(train) + [1] * len(test))
    adv_scores = cv_scores(
        pd.DataFrame(StandardScaler().fit_transform(Xadv), columns=FULL_22),
        pd.Series(yadv),
        lambda: LogisticRegression(max_iter=2000),
    )
    record("Adversarial validation (train-vs-test)", "22", adv_scores,
           note="~0.5 = no detectable covariate shift in P(X); says nothing about P(y|X)")

    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
