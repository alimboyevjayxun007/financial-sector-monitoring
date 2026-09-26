import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.data_loading import load_signals, load_transactions
from src.features import build as build_v1

def evaluate_subset(X, y, name=""):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42))
        ])
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = pipe.predict_proba(X.iloc[va_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[va_idx], p))
    mean_auc = np.mean(scores)
    std_auc = np.std(scores)
    print(f"[{name:<32}] {X.shape[1]} feats -> AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
    return mean_auc

def main():
    print("Loading data...")
    signals = load_signals(config.TRAIN_SIGNALS_PATH)
    txns = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
    y = signals[config.TARGET_COL]

    base_df = build_v1(signals, txns)
    X_base = base_df[config.FEATURE_COLUMNS].copy()
    evaluate_subset(X_base, y, "Baseline v1 (22 cols)")

    print("Computing candidate signals...")
    df = txns.merge(signals[[config.ID_COL, "signal_sanasi"]], on=config.ID_COL, how="inner")
    df = df[df["tranzaksiya_vaqti"] <= df["signal_sanasi"]].copy()
    df["days_before"] = (df["signal_sanasi"] - df["tranzaksiya_vaqti"]).dt.total_seconds() / 86400
    df["is_kirim"] = (df["kirim_chiqim"] == "kirim").astype(int)

    df_1d = df[df["days_before"] <= 1.0]
    agg_1d = df_1d.groupby(config.ID_COL).agg(
        amt_mean_1d=("miqdor_indeksi", "mean"),
        frac_kirim_1d=("is_kirim", "mean"),
    ).reset_index()

    df_7d = df[df["days_before"] <= 7.0]
    agg_7d = df_7d.groupby(config.ID_COL).agg(
        amt_mean_7d=("miqdor_indeksi", "mean"),
        amt_std_7d=("miqdor_indeksi", "std"),
        frac_kirim_7d=("is_kirim", "mean"),
    ).reset_index()

    flow = df.groupby([config.ID_COL, "kirim_chiqim"])["miqdor_indeksi"].sum().unstack(fill_value=0.0)
    flow["net_amt_flow"] = flow.get("kirim", 0.0) - flow.get("chiqim", 0.0)

    candidates = signals[[config.ID_COL]].merge(agg_1d, on=config.ID_COL, how="left")
    candidates = candidates.merge(agg_7d, on=config.ID_COL, how="left")
    candidates = candidates.merge(flow[["net_amt_flow"]], on=config.ID_COL, how="left")
    candidates = candidates.fillna(0.0)

    candidates["ratio_n_1d_to_7d"] = (X_base["n_txn_1d"]) / ((X_base["n_txn_7d"] / 7.0) + 0.1)

    test_cols = [
        "amt_mean_1d",
        "frac_kirim_1d",
        "amt_mean_7d",
        "amt_std_7d",
        "frac_kirim_7d",
        "net_amt_flow",
        "ratio_n_1d_to_7d"
    ]

    for col in test_cols:
        X_test = X_base.copy()
        X_test[col] = candidates[col]
        evaluate_subset(X_test, y, f"Base + {col}")

    X_combo = X_base.copy()
    X_combo["amt_mean_1d"] = candidates["amt_mean_1d"]
    X_combo["amt_std_7d"] = candidates["amt_std_7d"]
    evaluate_subset(X_combo, y, "Base + amt_mean_1d + amt_std_7d")

if __name__ == "__main__":
    main()
