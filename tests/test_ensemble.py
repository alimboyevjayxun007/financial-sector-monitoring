import numpy as np
import pandas as pd

from src.ensemble import choose, fit_weights, rank_average


def _oof(n=500):
    rng = np.random.default_rng(11)
    truth = rng.normal(size=n)
    y = pd.Series((truth > 0).astype(int))
    strong = truth + rng.normal(scale=0.5, size=n)
    weak = truth + rng.normal(scale=2.0, size=n)
    return {"strong": strong, "weak": weak}, y


def test_rank_average_returns_values_in_unit_interval():
    oof, _ = _oof()
    blended = rank_average(oof)
    assert blended.shape == (len(oof["strong"]),)
    assert blended.min() >= 0.0 and blended.max() <= 1.0


def test_rank_average_ignores_prediction_scale():
    oof, _ = _oof()
    scaled = {"strong": oof["strong"] * 1000 + 7, "weak": oof["weak"]}
    np.testing.assert_allclose(rank_average(oof), rank_average(scaled))


def test_fit_weights_favours_the_stronger_model():
    oof, y = _oof()
    weights = fit_weights(oof, y)
    assert weights["strong"] > weights["weak"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_choose_prefers_equal_weights_when_the_gain_is_small():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=400)
    y = pd.Series((truth > 0).astype(int))
    a = truth + rng.normal(scale=0.8, size=400)
    b = truth + rng.normal(scale=0.8, size=400)
    weights, method = choose({"a": a, "b": b}, y, min_gain=0.5)
    assert method == "equal"
    assert weights == {"a": 0.5, "b": 0.5}


def test_score_blend_matches_the_single_model_metric():
    """A blend must be scored per repeat then averaged, exactly as a single
    model is, so the two numbers are comparable."""
    from src.ensemble import score_blend

    rng = np.random.default_rng(7)
    truth = rng.normal(size=300)
    y = pd.Series((truth > 0).astype(int))
    per_repeat = {
        "a": np.stack([truth + rng.normal(scale=0.9, size=300) for _ in range(3)]),
        "b": np.stack([truth + rng.normal(scale=0.9, size=300) for _ in range(3)]),
    }
    mean, std, scores = score_blend(per_repeat, y, {"a": 0.5, "b": 0.5})
    assert len(scores) == 3
    assert mean == float(np.mean(scores))
    assert std == float(np.std(scores))
