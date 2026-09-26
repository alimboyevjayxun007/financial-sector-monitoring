import argparse
import logging
from pathlib import Path
from typing import Any, Sequence, Union

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_loading import load_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("src.predict")


def predict(model: Any, features_df: pd.DataFrame) -> pd.DataFrame:
    missing = set(config.FEATURE_COLUMNS) - set(features_df.columns)
    if missing:
        raise ValueError(f"features_df is missing required columns: {sorted(missing)}")
    feature_values = features_df[config.FEATURE_COLUMNS]
    if feature_values.isna().any().any():
        raise ValueError("features_df has NaN values in feature columns — check the feature pipeline")
    if not np.isfinite(feature_values.to_numpy(dtype=float)).all():
        raise ValueError("features_df has infinite (inf/-inf) values in feature columns")

    proba = model.predict_proba(feature_values)[:, 1]
    return pd.DataFrame({config.ID_COL: features_df[config.ID_COL], "ehtimollik": proba})


def validate_submission(
    df: pd.DataFrame,
    expected_ids: Union[pd.Series, Sequence[str], set],
    min_std: float = 0.01,
) -> None:
    assert list(df.columns) == [config.ID_COL, "ehtimollik"], "wrong columns/order"
    assert not df[config.ID_COL].duplicated().any(), "duplicate signal_id"

    pred_ids = set(df[config.ID_COL])
    exp_ids = set(expected_ids)
    diff = pred_ids ^ exp_ids
    assert not diff, f"signal_id set difference is not empty (diff count={len(diff)})"

    assert df["ehtimollik"].notna().all(), "missing predictions"
    assert df["ehtimollik"].between(0, 1).all(), "ehtimollik out of [0, 1]"

    if len(df) > 1:
        prob_min = float(df["ehtimollik"].min())
        prob_max = float(df["ehtimollik"].max())
        assert prob_min != prob_max, "ehtimollik values are constant (min == max); predictions collapsed"
        prob_std = float(df["ehtimollik"].std())
        assert prob_std >= min_std, f"ehtimollik std ({prob_std:.5f}) < {min_std}; distribution variance too low"


def write(predictions_df: pd.DataFrame, out_path: Union[str, Path]) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(target, index=False)
    logger.info("Submission successfully written to %s (%d rows)", target, len(predictions_df))
    return target


def run_predict(
    out_path: Union[str, Path],
    model_path: Path = config.MODEL_PATH,
    test_signals_path: Path = config.TEST_SIGNALS_PATH,
    test_features_path: Path = config.TEST_FEATURES_PATH,
) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run `python3 -m src.train` first."
        )
    if not test_features_path.exists():
        raise FileNotFoundError(
            f"{test_features_path} not found — run `python3 -m src.features` first."
        )

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Loading test signals from %s", test_signals_path)
    test_signals = load_signals(test_signals_path)

    logger.info("Loading test features from %s", test_features_path)
    test_features = pd.read_parquet(test_features_path)

    logger.info("Generating predictions for %d test signals...", len(test_features))
    predictions = predict(model, test_features)

    logger.info("Validating submission format...")
    validate_submission(predictions, test_signals[config.ID_COL])

    return write(predictions, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission predictions for Track B")
    parser.add_argument("--out", required=True, help="Destination CSV path, e.g. outputs/team_<TEAM_ID>.csv")
    args = parser.parse_args()

    run_predict(out_path=args.out)
