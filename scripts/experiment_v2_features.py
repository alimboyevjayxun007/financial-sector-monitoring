"""Feature exploration experiment: test candidate v2 features against the 0.5656 baseline.
Runs 5-fold Stratified CV with Logistic Regression (StandardScaler + class_weight='balanced').
"""
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

def evaluate_features(X, y, name="Feature Set"):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ])
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        probs = pipe.predict_proba(X.iloc[va_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[va_idx], probs))
    mean_auc = np.mean(scores)
    std_auc = np.std(scores)
    print(f"[{name}] {X.shape[1]} features -> CV ROC-AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
    return scores

def main():
    print("Loading raw train data...")
    signals = load_signals(config.TRAIN_SIGNALS_PATH)
    txns = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
    y = signals[config.TARGET_COL]

    print("\n1. Evaluating baseline v1 features (22 columns)...")
    base_df = build_v1(signals, txns)
    X_base = base_df[config.FEATURE_COLUMNS]
    base_scores = evaluate_features(X_base, y, "Baseline v1 (22 cols)")

    print("\n2. Computing candidate v2 features on raw transactions...")
    df = txns.merge(signals[[config.ID_COL, "signal_sanasi"]], on=config.ID_COL, how="inner")
    df = df[df["tranzaksiya_vaqti"] <= df["signal_sanasi"]].copy()
    df["days_before"] = (df["signal_sanasi"] - df["tranzaksiya_vaqti"]).dt.total_seconds() / 86400
    df["is_kirim"] = (df["kirim_chiqim"] == "kirim").astype(int)
    df["is_naqd"] = (df["tranzaksiya_turi"] == "naqd").astype(int)
    df["is_xalqaro"] = (df["tranzaksiya_turi"] == "xalqaro").astype(int)
    df["is_extreme"] = (df["miqdor_indeksi"].abs() > 2.0).astype(int)

    # Candidate Group A: Recency gap & last transaction proximity
    print("Computing Group A: Recency gap & Last transaction features...")
    recency_gap = df.groupby(config.ID_COL)["days_before"].min().rename("recency_days_min")
    recency_gap_df = signals[[config.ID_COL]].merge(recency_gap, on=config.ID_COL, how="left").fillna(999.0)

    # Candidate Group B: 7-day recent amounts & personal drift
    print("Computing Group B: 7-day amounts & personal drift...")
    df_7d = df[df["days_before"] <= 7.0]
    agg_7d = df_7d.groupby(config.ID_COL).agg(
        amt_mean_7d=("miqdor_indeksi", "mean"),
        amt_sum_7d=("miqdor_indeksi", "sum"),
        n_extreme_7d=("is_extreme", "sum"),
        n_xalqaro_7d=("is_xalqaro", "sum"),
        n_naqd_7d=("is_naqd", "sum"),
    ).reset_index()
    agg_7d_df = signals[[config.ID_COL]].merge(agg_7d, on=config.ID_COL, how="left").fillna(0.0)

    # Candidate Group C: 1-day amounts
    print("Computing Group C: 1-day amounts...")
    df_1d = df[df["days_before"] <= 1.0]
    agg_1d = df_1d.groupby(config.ID_COL).agg(
        amt_mean_1d=("miqdor_indeksi", "mean"),
        amt_sum_1d=("miqdor_indeksi", "sum"),
    ).reset_index()
    agg_1d_df = signals[[config.ID_COL]].merge(agg_1d, on=config.ID_COL, how="left").fillna(0.0)

    # Merge candidate features with base
    X_exp = X_base.copy()
    X_exp["recency_days_min"] = recency_gap_df["recency_days_min"]
    X_exp["amt_mean_7d"] = agg_7d_df["amt_mean_7d"]
    X_exp["amt_sum_7d"] = agg_7d_df["amt_sum_7d"]
    X_exp["n_extreme_7d"] = agg_7d_df["n_extreme_7d"]
    X_exp["n_xalqaro_7d"] = agg_7d_df["n_xalqaro_7d"]
    X_exp["n_naqd_7d"] = agg_7d_df["n_naqd_7d"]
    X_exp["amt_mean_1d"] = agg_1d_df["amt_mean_1d"]
    X_exp["amt_sum_1d"] = agg_1d_df["amt_sum_1d"]

    # Personal drift from overall mean
    X_exp["drift_amt_mean_7d"] = X_exp["amt_mean_7d"] - X_exp["amt_mean"]
    X_exp["ratio_velocity_7d"] = (X_exp["n_txn_7d"] / 7.0) / (X_exp["velocity"] + 0.01)

    print("\n3. Testing individual candidate features addition...")
    for col in [
        "recency_days_min", "amt_mean_7d", "amt_sum_7d", "n_extreme_7d",
        "n_xalqaro_7d", "n_naqd_7d", "amt_mean_1d", "amt_sum_1d",
        "drift_amt_mean_7d", "ratio_velocity_7d"
    ]:
        X_single = X_base.copy()
        X_single[col] = X_exp[col]
        evaluate_features(X_single, y, f"Base + {col}")

    print("\n4. Testing combined candidate set...")
    evaluate_features(X_exp, y, f"All Candidates ({X_exp.shape[1]} cols)")

if __name__ == "__main__":
    main()
