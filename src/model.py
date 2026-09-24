"""Track B: ModelTrainer.

Model choice, backed by CV experiments (see README.md "Model tanlash"):
plain regularized Logistic Regression beat HistGradientBoostingClassifier
(0.563 vs 0.539 mean CV ROC-AUC) and degree-2 polynomial interactions made
things worse (0.547) on this feature set. With ~14k rows and every raw
feature correlating with the target at |r| <= 0.06 (see EDA), gradient-
boosted trees have enough capacity to fit noise instead of signal; a
regularized linear model does not. C was swept 0.01-3.0 with no meaningful
change in CV AUC, so the default is used rather than tuning further.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _make_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def train(features_df: pd.DataFrame, target: pd.Series) -> Pipeline:
    model = _make_classifier()
    model.fit(features_df, target)
    return model


def cross_validate(features_df: pd.DataFrame, target: pd.Series, n_splits: int = 5) -> float:
    """StratifiedKFold CV (keeps the ~17% positive rate in every fold),
    returns the mean out-of-fold ROC-AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = features_df.reset_index(drop=True)
    y = target.reset_index(drop=True)

    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        model = _make_classifier()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[val_idx], proba))

    return float(np.mean(aucs))
