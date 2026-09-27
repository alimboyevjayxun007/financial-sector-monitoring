"""Precompute small aggregates so the website ships without the raw data."""
import numpy as np
import pandas as pd

from src.config import DIRECTIONS, TX_TYPES

OUTCOME = {0: "Dismissed", 1: "Escalated"}


def compute(signals: pd.DataFrame, tx: pd.DataFrame) -> dict[str, pd.DataFrame]:
    labelled = tx.merge(signals[["signal_id", "eskalatsiya"]], on="signal_id", how="inner")
    per_signal = labelled.groupby(["signal_id", "eskalatsiya"]).size().rename("n").reset_index()

    tables: dict[str, pd.DataFrame] = {}

    tables["target"] = (
        signals["eskalatsiya"].map(OUTCOME).value_counts().rename_axis("outcome").rename("alerts").reset_index()
    )
    tables["weekly_volume"] = (
        tx.set_index("tranzaksiya_vaqti").resample("W").size().rename("transactions").reset_index()
    )
    tables["direction"] = (
        tx["kirim_chiqim"].value_counts().reindex(DIRECTIONS).rename_axis("direction").rename("transactions").reset_index()
    )
    tables["types"] = (
        tx["tranzaksiya_turi"].value_counts().reindex(TX_TYPES).rename_axis("type").rename("transactions").reset_index()
    )
    tables["amount_by_outcome"] = (
        labelled.groupby("eskalatsiya")["miqdor_indeksi"]
        .agg(["count", "mean", "median", "std"])
        .rename(index=OUTCOME)
        .rename_axis("outcome")
        .reset_index()
    )
    tables["volume_by_outcome"] = (
        per_signal.groupby("eskalatsiya")["n"]
        .agg(["mean", "median", "min", "max"])
        .rename(index=OUTCOME)
        .rename_axis("outcome")
        .reset_index()
    )
    bins = np.arange(0, 190, 10)
    binned = pd.cut(labelled["days_before"], bins=bins, right=False, labels=bins[:-1])
    tables["days_before_hist"] = (
        labelled.assign(bucket=binned.astype("float64"))
        .groupby(["bucket", "eskalatsiya"], observed=True)
        .size()
        .rename("transactions")
        .reset_index()
        .replace({"eskalatsiya": OUTCOME})
    )
    tables["type_by_outcome"] = (
        labelled.groupby(["tranzaksiya_turi", "eskalatsiya"])
        .size()
        .rename("transactions")
        .reset_index()
        .replace({"eskalatsiya": OUTCOME})
    )
    return tables
