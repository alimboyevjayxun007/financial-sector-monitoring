import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ID, OUTPUTS, PROBA, SUBMISSION_NAME


class SubmissionError(ValueError):
    """Raised when a submission frame violates a competition rule."""


def check(frame: pd.DataFrame, expected_ids: pd.Series) -> list[tuple[str, bool, str]]:
    expected = set(expected_ids)
    actual = list(frame[ID]) if ID in frame.columns else []
    probabilities = frame[PROBA] if PROBA in frame.columns else pd.Series(dtype=float)

    missing = expected - set(actual)
    extra = set(actual) - expected
    duplicates = pd.Series(actual).duplicated().sum()
    out_of_range = (~probabilities.between(0.0, 1.0)).sum() if len(probabilities) else 0
    null_count = probabilities.isna().sum() if len(probabilities) else 0
    distinct = probabilities.nunique(dropna=True) if len(probabilities) else 0

    return [
        ("columns are exactly [signal_id, ehtimollik]", list(frame.columns) == [ID, PROBA],
         f"got {list(frame.columns)}"),
        ("row count matches test signals", len(frame) == len(expected_ids),
         f"{len(frame)} rows vs {len(expected_ids)} expected"),
        ("no missing IDs", not missing, f"{len(missing)} missing"),
        ("no extra IDs", not extra, f"{len(extra)} extra"),
        ("no duplicate IDs", duplicates == 0, f"{duplicates} duplicated"),
        ("no null probabilities", null_count == 0, f"{null_count} null"),
        ("all probabilities within [0, 1]", out_of_range == 0, f"{out_of_range} out of range"),
        # The organizer's sample_submission is 6000 identical values and would
        # otherwise pass every rule above with a clean report.
        ("predictions are not a constant placeholder", distinct > 1,
         f"{distinct} distinct value(s)"),
    ]


def write(
    frame: pd.DataFrame,
    expected_ids: pd.Series,
    path: Path | None = None,
) -> Path:
    ordered = (
        expected_ids.to_frame(name=ID)
        .merge(frame[[ID, PROBA]], on=ID, how="left")
        .reset_index(drop=True)
    )
    results = check(ordered, expected_ids)
    for rule, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {rule} ({detail})")
    failures = [rule for rule, passed, _ in results if not passed]
    if failures:
        raise SubmissionError("Submission rules failed: " + "; ".join(failures))

    target = (OUTPUTS / SUBMISSION_NAME) if path is None else path
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so a failure mid-serialization cannot leave a partial
    # CSV where a complete one is expected.
    staging = target.with_suffix(target.suffix + ".part")
    try:
        ordered.to_csv(staging, index=False)
        os.replace(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target
