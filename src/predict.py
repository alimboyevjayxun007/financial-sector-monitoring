"""Track B: Predictor + SubmissionWriter."""
import argparse
import logging
from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_loading import load_signals

# Setup module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("src.predict")


def predict(model: Any, features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate calibrated escalation probabilities for given feature set.

    Args:
        model: Fitted classifier supporting predict_proba (e.g. CalibratedClassifierCV).
        features_df: DataFrame containing required columns defined in config.FEATURE_COLUMNS.

    Returns:
        DataFrame containing signal_id and predicted probability column 'ehtimollik'.

    Raises:
        ValueError: If required columns are missing, or contain NaN/infinite values.
    """
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
    """Validate that submission strictly conforms to competition format and distribution rules.

    Checks:
        1. Exact column names and ordering: [signal_id, ehtimollik]
        2. No duplicate signal_id values
        3. Zero set difference between predictions and expected IDs (no missing, no unexpected)
        4. No NaN/null probability values
        5. Probabilities bounded in [0, 1]
        6. Distribution spread: probabilities must not collapse to a single constant (min != max)
        7. Distribution variance: standard deviation must meet minimum diversity threshold (std >= min_std)

    Args:
        df: Predictions DataFrame to validate.
        expected_ids: Series or collection of expected signal_id values.
        min_std: Minimum allowed standard deviation (default 0.01) to protect against collapsed predictions.

    Raises:
        AssertionError: If any competition format or distribution invariant is violated.
    """
    assert list(df.columns) == [config.ID_COL, "ehtimollik"], "wrong columns/order"
    assert not df[config.ID_COL].duplicated().any(), "duplicate signal_id"

    # Symmetric set difference check
    pred_ids = set(df[config.ID_COL])
    exp_ids = set(expected_ids)
    diff = pred_ids ^ exp_ids
    assert not diff, f"signal_id set difference is not empty (diff count={len(diff)})"

    assert df["ehtimollik"].notna().all(), "missing predictions"
    assert df["ehtimollik"].between(0, 1).all(), "ehtimollik out of [0, 1]"

    # Distribution checks (guards against collapsed or constant prediction outputs)
    if len(df) > 1:
        prob_min = float(df["ehtimollik"].min())
        prob_max = float(df["ehtimollik"].max())
        assert prob_min != prob_max, "ehtimollik values are constant (min == max); predictions collapsed"
        prob_std = float(df["ehtimollik"].std())
        assert prob_std >= min_std, f"ehtimollik std ({prob_std:.5f}) < {min_std}; distribution variance too low"


def write(predictions_df: pd.DataFrame, out_path: Union[str, Path]) -> Path:
    """Write validated submission predictions to a CSV file.

    Args:
        predictions_df: DataFrame containing signal_id and ehtimollik.
        out_path: Destination path for the CSV output.

    Returns:
        Path object pointing to the written CSV file.
    """
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
    """Load model and test features, generate predictions, validate format, and write output.

    Args:
        out_path: Destination path for the submission CSV file.
        model_path: Path to serialized model artifact.
        test_signals_path: Path to raw test signals CSV.
        test_features_path: Path to processed test features parquet file.

    Returns:
        Path to the generated submission file.

    Raises:
        FileNotFoundError: If model or feature files do not exist.
    """
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

