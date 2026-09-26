import logging
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src import config
from src.model import cross_validate, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("src.train")


def load_train_features(path: Path = config.TRAIN_FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python3 -m src.features` "
            "first to generate it. (There used to be a silent fallback to random dummy "
            "features here for early-development bootstrapping; it was removed because "
            "training a real model on noise and saving it as model.pkl with no error is "
            "exactly the kind of silent-wrong-output failure a production pipeline must "
            "not allow — see README.md 'Production-readiness' notes.)"
        )
    logger.info("Loading training features from %s", path)
    return pd.read_parquet(path)


def run_train(
    save_path: Path = config.MODEL_PATH,
) -> Tuple[CalibratedClassifierCV, float]:
    df = load_train_features()
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COL]

    logger.info("Running 5-fold Stratified CV evaluation...")
    auc = cross_validate(X, y)
    logger.info("CV ROC-AUC: %.4f", auc)

    logger.info("Fitting final model on full training set...")
    model_obj = train(X, y)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_obj, save_path)
    logger.info("Model successfully saved to %s", save_path)

    return model_obj, auc


if __name__ == "__main__":
    run_train()
