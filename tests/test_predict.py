import numpy as np
import pandas as pd

from src import config
from src.model import train
from src.predict import predict, validate_submission


def _fake_features(ids):
    n = len(ids)
    rng = np.random.default_rng(0)
    data = {col: rng.normal(size=n) for col in config.FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    df.insert(0, config.ID_COL, ids)
    return df


def test_predict_returns_two_columns_in_order():
    ids = [f"SG_{i}" for i in range(20)]
    train_df = _fake_features(ids)
    y = pd.Series(([0] * 16) + ([1] * 4))
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


def test_real_train_and_predict_end_to_end_if_features_available():
    if not config.TRAIN_FEATURES_PATH.exists() or not config.TEST_FEATURES_PATH.exists():
        return
    train_features = pd.read_parquet(config.TRAIN_FEATURES_PATH)
    test_features = pd.read_parquet(config.TEST_FEATURES_PATH)

    model = train(train_features[config.FEATURE_COLUMNS], train_features[config.TARGET_COL])
    preds = predict(model, test_features)

    validate_submission(preds, test_features[config.ID_COL])
