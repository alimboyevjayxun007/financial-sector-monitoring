import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import EXPERIMENTS, SEED

LOG_PATH = EXPERIMENTS / "log.csv"
N_SPLITS = 5


@dataclass(frozen=True)
class CVResult:
    mean: float
    std: float
    oof: np.ndarray
    oof_per_repeat: np.ndarray
    repeat_scores: tuple[float, ...]

    @property
    def score(self) -> float:
        return self.mean - self.std


def _impute(fit_frame: pd.DataFrame, apply_frames: Sequence[pd.DataFrame]):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(fit_frame)
    return [
        pd.DataFrame(imputer.transform(frame), index=frame.index, columns=frame.columns)
        for frame in apply_frames
    ]


def evaluate(
    model_factory: Callable[[int], object],
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 4,
) -> CVResult:
    """Repeated stratified CV. Returns per-repeat mean/std and averaged OOF."""
    y = pd.Series(np.asarray(y), index=X.index)
    per_repeat = np.zeros((n_repeats, len(X)))
    repeat_scores: list[float] = []

    for repeat in range(n_repeats):
        seed = SEED + repeat
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))
        for train_idx, valid_idx in splitter.split(X, y):
            X_fit_raw = X.iloc[train_idx]
            X_val_raw = X.iloc[valid_idx]
            X_fit, X_val = _impute(X_fit_raw, [X_fit_raw, X_val_raw])
            model = model_factory(seed)
            model.fit(X_fit, y.iloc[train_idx])
            oof[valid_idx] = model.predict_proba(X_val)[:, 1]
        repeat_scores.append(roc_auc_score(y, oof))
        per_repeat[repeat] = oof

    return CVResult(
        mean=float(np.mean(repeat_scores)),
        std=float(np.std(repeat_scores)),
        oof=per_repeat.mean(axis=0),
        oof_per_repeat=per_repeat,
        repeat_scores=tuple(repeat_scores),
    )


def log_run(
    result: CVResult,
    *,
    families: Sequence[str],
    n_columns: int,
    model: str,
    params: dict,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "families": "|".join(families),
                "n_columns": n_columns,
                "model": model,
                "params": json.dumps(params, sort_keys=True, default=str),
                "cv_mean": round(result.mean, 6),
                "cv_std": round(result.std, 6),
                "score": round(result.score, 6),
            }
        ]
    )
    header = not LOG_PATH.exists()
    row.to_csv(LOG_PATH, mode="a", header=header, index=False)
