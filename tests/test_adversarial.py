import numpy as np
import pandas as pd

from src.adversarial import drift_report


def test_identical_distributions_are_indistinguishable():
    rng = np.random.default_rng(13)
    X_train = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    X_test = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    auc, importance = drift_report(X_train, X_test)
    assert 0.40 < auc < 0.60
    assert list(importance.index) == ["a", "b"] or set(importance.index) == {"a", "b"}


def test_a_shifted_column_is_detected_and_ranked_first():
    rng = np.random.default_rng(14)
    X_train = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    X_test = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500) + 8})
    auc, importance = drift_report(X_train, X_test)
    assert auc > 0.85
    assert importance.index[0] == "b"
