"""Keep the test suite out of the project's real experiment log.

select_families() and tune() call log_run() internally, so without this the
suite appends synthetic rows to experiments/log.csv and corrupts the record
the EDA site and notebook read from. The prediction fixtures likewise write a
submission CSV, which must never land in the real outputs/ directory.
"""
import pytest

import src.selection as selection
import src.submission as submission
import src.validation as validation


@pytest.fixture(autouse=True, scope="session")
def _redirect_experiment_artifacts(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("experiments")
    validation.LOG_PATH = scratch / "log.csv"
    selection.SELECTION_PATH = scratch / "selected_features.json"
    # submission.py binds OUTPUTS at import time; without this a fixture run
    # writes a synthetic team_2ABB3C78.csv into the real outputs/ directory.
    submission.OUTPUTS = scratch / "outputs"
    yield
