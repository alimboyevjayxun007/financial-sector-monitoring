import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import SEED
from src.models import make_factory


def drift_report(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[float, pd.Series]:
    """Train a classifier to separate train from test. AUC near 0.5 means no drift."""
    columns = [c for c in X_train.columns if c in X_test.columns]
    combined = pd.concat([X_train[columns], X_test[columns]], ignore_index=True)
    is_test = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = pd.DataFrame(imputer.fit_transform(combined), columns=columns)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(imputed))
    importances = np.zeros(len(columns))
    for train_idx, valid_idx in splitter.split(imputed, is_test):
        model = make_factory("lightgbm")(SEED)
        model.fit(imputed.iloc[train_idx], is_test[train_idx])
        oof[valid_idx] = model.predict_proba(imputed.iloc[valid_idx])[:, 1]
        importances += model.feature_importances_

    auc = float(roc_auc_score(is_test, oof))
    ranked = pd.Series(importances, index=columns).sort_values(ascending=False)
    return auc, ranked
