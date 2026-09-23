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
    with pytest.raises(AssertionError):
        validate_submission(df, ids)
