import numpy as np
import pandas as pd

from src.selection import eliminate_columns, load_selection, save_selection, select_families


def _dataset(n=600):
    rng = np.random.default_rng(3)
    signal = rng.normal(size=n)
    y = pd.Series((signal + rng.normal(scale=0.6, size=n) > 0).astype(int))
    useful = pd.DataFrame({"good": signal}, index=range(n))
    noise = pd.DataFrame({f"noise{i}": rng.normal(size=n) for i in range(6)}, index=range(n))
    return {"useful": useful, "noise": noise}, y


def test_select_families_keeps_the_informative_family():
    by_family, y = _dataset()
    chosen, history = select_families(by_family, y, model="logreg")
    assert "useful" in chosen


def test_select_families_returns_history_rows_with_scores():
    by_family, y = _dataset()
    chosen, history = select_families(by_family, y, model="logreg")
    assert history
    assert {"family", "score", "accepted"} <= set(history[0])


def test_eliminate_columns_returns_a_non_empty_subset():
    by_family, y = _dataset()
    X = pd.concat(by_family.values(), axis=1)
    kept = eliminate_columns(X, y, model="logreg")
    assert kept
    assert set(kept) <= set(X.columns)


def test_save_and_load_selection_round_trip(tmp_path, monkeypatch):
    import src.selection as selection

    monkeypatch.setattr(selection, "SELECTION_PATH", tmp_path / "selected.json")
    save_selection(["a", "b"], ["base"])
    loaded = load_selection()
    assert loaded == {"columns": ["a", "b"], "families": ["base"]}


def test_global_column_selection_finds_the_informative_column_across_families():
    """Family-level greedy search discards every column of a rejected family.
    Ranking columns globally keeps a good column from a bad family."""
    from src.selection import select_columns_globally

    rng = np.random.default_rng(17)
    n = 800
    signal = rng.normal(size=n)
    y = pd.Series((signal + rng.normal(scale=0.7, size=n) > 0).astype(int))
    X = pd.DataFrame({"strong": signal + rng.normal(scale=0.4, size=n)}, index=range(n))
    # A single useful column buried among noise, as if from a rejected family.
    X["buried"] = signal + rng.normal(scale=1.0, size=n)
    for i in range(20):
        X[f"noise{i}"] = rng.normal(size=n)

    columns, trace = select_columns_globally(X, y, model="logreg", sizes=(2, 5, 10))
    assert "strong" in columns
    assert len(columns) < X.shape[1]
    assert trace
    assert {"size", "mean", "std", "score"} <= set(trace[0])


def test_global_column_selection_returns_the_best_scoring_size():
    from src.selection import select_columns_globally

    rng = np.random.default_rng(18)
    n = 600
    signal = rng.normal(size=n)
    y = pd.Series((signal > 0).astype(int))
    X = pd.DataFrame({"a": signal + rng.normal(scale=0.5, size=n)}, index=range(n))
    for i in range(10):
        X[f"n{i}"] = rng.normal(size=n)

    columns, trace = select_columns_globally(X, y, model="logreg", sizes=(2, 4, 8))
    best = max(trace, key=lambda row: row["score"])
    assert len(columns) == best["size"]
