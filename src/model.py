"""Track B mission: ModelTrainer.

TODO(track-b): implement train() and cross_validate(). Suggested starting
point: LGBMClassifier with class_weight="balanced" (target is ~17%
positive), evaluated with StratifiedKFold + roc_auc_score.
"""
import pandas as pd


def train(features_df: pd.DataFrame, target: pd.Series):
    raise NotImplementedError("Track B: fit a classifier and return it")


def cross_validate(features_df: pd.DataFrame, target: pd.Series, n_splits: int = 5) -> float:
    raise NotImplementedError(
        "Track B: run StratifiedKFold CV, return mean ROC-AUC across folds"
    )
