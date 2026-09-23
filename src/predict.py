"""Track B: Predictor + SubmissionWriter."""
import argparse

import pandas as pd

from src import config
from src.data_loading import load_signals


def predict(model, features_df: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(features_df[config.FEATURE_COLUMNS])[:, 1]
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
    if config.TEST_FEATURES_PATH.exists():
        test_features = pd.read_parquet(config.TEST_FEATURES_PATH)
    else:
        print(f"[predict] {config.TEST_FEATURES_PATH} not found yet, using dummy features")
        test_features = config.make_dummy_features(test_signals)

    predictions = predict(model, test_features)
    validate_submission(predictions, test_signals[config.ID_COL])
    write(predictions, args.out)
    print(f"wrote {args.out}")
