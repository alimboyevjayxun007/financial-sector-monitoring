import pandas as pd
import pytest

from src.predict import validate_submission


def test_validate_submission_ok():
    ids = pd.Series(["SG_1", "SG_2", "SG_3"])
    df = pd.DataFrame({"signal_id": ids, "ehtimollik": [0.1, 0.9, 0.5]})
    validate_submission(df, ids)


def test_validate_submission_rejects_out_of_range():
    ids = pd.Series(["SG_1", "SG_2"])
    df = pd.DataFrame({"signal_id": ids, "ehtimollik": [0.1, 1.5]})
    with pytest.raises(AssertionError):
        validate_submission(df, ids)


def test_validate_submission_rejects_duplicates():
    ids = pd.Series(["SG_1", "SG_1"])
    df = pd.DataFrame({"signal_id": ids, "ehtimollik": [0.1, 0.2]})
    with pytest.raises(AssertionError):
        validate_submission(df, pd.Series(["SG_1"]))


def test_validate_submission_rejects_missing_id():
    ids = pd.Series(["SG_1", "SG_2"])
    df = pd.DataFrame({"signal_id": ["SG_1"], "ehtimollik": [0.1]})
    with pytest.raises(AssertionError, match="set difference"):
        validate_submission(df, ids)


def test_validate_submission_rejects_unexpected_extra_ids():
    ids = pd.Series(["SG_1", "SG_2"])
    df = pd.DataFrame({"signal_id": ["SG_1", "SG_2", "SG_EXTRA"], "ehtimollik": [0.1, 0.5, 0.9]})
    with pytest.raises(AssertionError, match="set difference"):
        validate_submission(df, ids)


def test_validate_submission_rejects_constant_probabilities():
    ids = pd.Series(["SG_1", "SG_2", "SG_3"])
    df = pd.DataFrame({"signal_id": ids, "ehtimollik": [0.5, 0.5, 0.5]})
    with pytest.raises(AssertionError, match="constant"):
        validate_submission(df, ids)


def test_validate_submission_rejects_low_variance_distribution():
    ids = pd.Series([f"SG_{i}" for i in range(10)])
    df = pd.DataFrame({"signal_id": ids, "ehtimollik": [0.17 + (0.0001 * (i % 2)) for i in range(10)]})
    with pytest.raises(AssertionError, match="variance too low"):
        validate_submission(df, ids, min_std=0.01)


def test_validate_submission_on_real_submission_file_if_available():
    from src import config
    sub_path = config.OUTPUTS_DIR / "team_C6FD20A0.csv"
    if not sub_path.exists() or not config.TEST_SIGNALS_PATH.exists():
        pytest.skip("outputs/team_C6FD20A0.csv or test_signals.csv not present")
    
    sub_df = pd.read_csv(sub_path)
    test_signals = pd.read_csv(config.TEST_SIGNALS_PATH)
    validate_submission(sub_df, test_signals[config.ID_COL])
