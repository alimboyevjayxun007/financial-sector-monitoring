"""Track B: Predictor + SubmissionWriter."""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.data_loading import load_signals


def predict(model, features_df: pd.DataFrame) -> pd.DataFrame:
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


def validate_submission(df: pd.DataFrame, expected_ids: pd.Series) -> None:
    assert list(df.columns) == [config.ID_COL, "ehtimollik"], "wrong columns/order"
    assert not df[config.ID_COL].duplicated().any(), "duplicate signal_id"
    assert set(df[config.ID_COL]) == set(expected_ids), "missing/unknown signal_id"
    assert df["ehtimollik"].notna().all(), "missing predictions"
    assert df["ehtimollik"].between(0, 1).all(), "ehtimollik out of [0, 1]"


def write(predictions_df: pd.DataFrame, out_path: str) -> None:
    predictions_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="e.g. outputs/team_<TEAM_ID>.csv")
    args = parser.parse_args()

    import joblib

    model = joblib.load(config.OUTPUTS_DIR / "model.pkl")

    test_signals = load_signals(config.TEST_SIGNALS_PATH)
    if not config.TEST_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{config.TEST_FEATURES_PATH} not found — run `python3 -m src.features` first."
        )
    test_features = pd.read_parquet(config.TEST_FEATURES_PATH)

    predictions = predict(model, test_features)
    validate_submission(predictions, test_signals[config.ID_COL])
    write(predictions, args.out)
    print(f"wrote {args.out}")
