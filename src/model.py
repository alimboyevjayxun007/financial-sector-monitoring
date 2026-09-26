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
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def train(features_df: pd.DataFrame, target: pd.Series) -> CalibratedClassifierCV:
    model = _make_classifier()
    model.fit(features_df, target)
    return model


def cross_validate(features_df: pd.DataFrame, target: pd.Series, n_splits: int = 5) -> float:
    if features_df is None or len(features_df) == 0:
        raise ValueError("features_df must not be empty")
    if target is None or len(target) == 0:
        raise ValueError("target must not be empty")
    if len(features_df) != len(target):
        raise ValueError(f"Length mismatch: features_df has {len(features_df)} rows but target has {len(target)} rows")

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
