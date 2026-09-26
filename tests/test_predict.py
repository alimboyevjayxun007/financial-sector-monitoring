import numpy as np
import pandas as pd
import pytest

from src import config
from src.model import train
from src.predict import predict, validate_submission, write


def _fake_features(ids):
    n = len(ids)
    rng = np.random.default_rng(0)
    data = {col: rng.normal(size=n) for col in config.FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    df.insert(0, config.ID_COL, ids)
    return df


def test_predict_returns_two_columns_in_order():
    ids = [f"SG_{i}" for i in range(100)]
    train_df = _fake_features(ids)
    y = pd.Series(([0] * 80) + ([1] * 20))
    model = train(train_df[config.FEATURE_COLUMNS], y)

    test_df = _fake_features(ids)
    preds = predict(model, test_df)

    assert list(preds.columns) == [config.ID_COL, "ehtimollik"]
    assert len(preds) == len(ids)
    validate_submission(preds, pd.Series(ids))


def test_predict_probabilities_within_unit_range():
    ids = [f"SG_{i}" for i in range(50)]
    train_df = _fake_features(ids)
    y = pd.Series(([0] * 40) + ([1] * 10))
    model = train(train_df[config.FEATURE_COLUMNS], y)

    test_df = _fake_features([f"SG_test_{i}" for i in range(15)])
    preds = predict(model, test_df)

    assert preds["ehtimollik"].between(0, 1).all()
    assert preds["ehtimollik"].notna().all()


def test_predict_rejects_missing_feature_columns():
    ids = [f"SG_{i}" for i in range(100)]
    train_df = _fake_features(ids)
    y = pd.Series(([0] * 80) + ([1] * 20))
    model = train(train_df[config.FEATURE_COLUMNS], y)

    incomplete = _fake_features(ids).drop(columns=[config.FEATURE_COLUMNS[0]])
    with pytest.raises(ValueError, match="missing required columns"):
        predict(model, incomplete)


def test_predict_rejects_nan_in_features():
    ids = [f"SG_{i}" for i in range(100)]
    train_df = _fake_features(ids)
    y = pd.Series(([0] * 80) + ([1] * 20))
    model = train(train_df[config.FEATURE_COLUMNS], y)

    with_nan = _fake_features(ids)
    with_nan.loc[0, config.FEATURE_COLUMNS[0]] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        predict(model, with_nan)


def test_real_train_and_predict_end_to_end_if_features_available():
    if not config.TRAIN_FEATURES_PATH.exists() or not config.TEST_FEATURES_PATH.exists():
        pytest.skip(
            "data/processed/*.parquet not present (gitignored) — run "
            "`python3 -m src.features` first to exercise this real-data test"
        )
    train_features = pd.read_parquet(config.TRAIN_FEATURES_PATH)
    test_features = pd.read_parquet(config.TEST_FEATURES_PATH)

    model = train(train_features[config.FEATURE_COLUMNS], train_features[config.TARGET_COL])
    preds = predict(model, test_features)

    validate_submission(preds, test_features[config.ID_COL])


def test_write_produces_valid_submission_file(tmp_path):
    out_file = tmp_path / "team_test.csv"
    preds = pd.DataFrame({
        config.ID_COL: [f"SG_{i:04d}" for i in range(25)],
        "ehtimollik": np.linspace(0.05, 0.95, 25),
    })
    write(preds, str(out_file))

    assert out_file.exists()
    loaded = pd.read_csv(out_file)
    assert list(loaded.columns) == [config.ID_COL, "ehtimollik"]
    assert len(loaded) == 25
    assert not loaded[config.ID_COL].duplicated().any()
    assert (loaded["ehtimollik"] >= 0).all() and (loaded["ehtimollik"] <= 1).all()


def test_regression_no_make_dummy_features_in_config():
    assert not hasattr(config, "make_dummy_features"), (
        "make_dummy_features must NOT exist in src.config (vulnerability guard)"
    )


def test_inspect_signal_tool_runs_without_error():
    from scripts import inspect_signal
    model, train_df, test_df, mean_coef, mean_center, mean_scale = inspect_signal.load_resources()
    active_df = test_df if test_df is not None else train_df
    if active_df is not None:
        sample_id = active_df.iloc[0][config.ID_COL]
        inspect_signal.inspect_single_signal(sample_id, active_df, model, mean_coef, mean_center, mean_scale)
