"""Track B: ModelTrainer.

Model choice, backed by `scripts/model_selection_experiments.py` (run it to
reproduce every number below -- see README.md "Model tanlash" 5.1/5.1.1/5.2
for the full writeup and the honest statistical caveats):

- Logistic Regression beat HistGradientBoostingClassifier, RandomForest,
  SVC, MLP, and a tuned LightGBM (15-config randomized search) on the base
  18-feature set: 0.5629 vs 0.5396-0.5611 5-fold CV ROC-AUC. With ~14k rows
  and every raw feature correlating with the target at |r| <= 0.06 (see
  EDA), gradient-boosted trees have enough capacity to fit noise instead of
  signal; a regularized linear model does not.
- Adding 4 hour/day-of-week entropy features (18 -> 22) moved CV AUC from
  0.5629 to 0.5656 (5-fold) / 0.5652 to 0.5677 (5x5 repeat) -- a real but
  NOT statistically significant shift once measured properly: the
  Nadeau-Bengio corrected paired t-test (which accounts for the
  non-independence of repeated-CV folds -- a naive paired t-test here is
  anti-conservative and gives a misleadingly small p-value) gives p=0.17.
  Kept anyway because the features are domain-motivated, cheap, and the
  effect direction was consistently positive, not because it's a proven
  win -- do not oversell this.
- The raw `predict_proba` output is poorly calibrated: `class_weight=
  "balanced"` pushes the mean predicted probability to ~0.49 vs the true
  ~17.2% base rate. Since the competition asks for a column literally named
  "ehtimollik" (probability), this matters even though ROC-AUC itself is
  threshold/calibration-invariant. `CalibratedClassifierCV` (Platt/sigmoid
  scaling) fixes this -- mean predicted probability lands at 0.172, almost
  exactly the true base rate -- for a negligible AUC cost (0.5638 vs
  0.5656 on the 22-feature set, within noise).
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def _make_classifier() -> CalibratedClassifierCV:
    base = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced", random_state=config.RANDOM_SEED)),
        ]
    )
    # cv=5 internally cross-validates the calibration mapping so it isn't
    # fit on the same rows used to fit the base classifier.
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def train(features_df: pd.DataFrame, target: pd.Series) -> CalibratedClassifierCV:
    model = _make_classifier()
    model.fit(features_df, target)
    return model


def cross_validate(features_df: pd.DataFrame, target: pd.Series, n_splits: int = 5) -> float:
    """StratifiedKFold CV (keeps the ~17% positive rate in every fold),
    returns the mean out-of-fold ROC-AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    X = features_df.reset_index(drop=True)
    y = target.reset_index(drop=True)

    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        model = _make_classifier()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[val_idx], proba))

    return float(np.mean(aucs))
