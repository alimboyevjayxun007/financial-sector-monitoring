"""The test suite must never write into the project's experiment log."""
from src import config
import src.validation as validation


def test_experiment_log_is_redirected_away_from_the_repository():
    assert validation.LOG_PATH != config.EXPERIMENTS / "log.csv"
    assert config.EXPERIMENTS not in validation.LOG_PATH.parents


def test_the_suite_cannot_write_the_real_submission_file():
    """A fixture run once wrote a synthetic team_2ABB3C78.csv into the real
    outputs/ directory, where it could have been submitted by mistake."""
    import src.submission as submission

    assert submission.OUTPUTS != config.OUTPUTS
