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
from src.features import build as build_current

def main():
    print("Loading data...")
    signals = load_signals(config.TRAIN_SIGNALS_PATH)
    txns = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
    y = signals[config.TARGET_COL]

    print("Building current 25 features...")
    current_df = build_current(signals, txns)
    X_curr = current_df[config.FEATURE_COLUMNS].copy()

    # Pre-merge for additional feature calculations
    df = txns.merge(signals[[config.ID_COL, "signal_sanasi"]], on=config.ID_COL, how="inner")
    df = df[df["tranzaksiya_vaqti"] <= df["signal_sanasi"]].copy()
    df["days_before"] = (df["signal_sanasi"] - df["tranzaksiya_vaqti"]).dt.total_seconds() / 86400
    df["is_kirim"] = (df["kirim_chiqim"] == "kirim").astype(int)
    df["is_chiqim"] = (df["kirim_chiqim"] == "chiqim").astype(int)
    df["is_naqd"] = (df["tranzaksiya_turi"] == "naqd").astype(int)
    df["hour"] = df["tranzaksiya_vaqti"].dt.hour
    df["is_dark_hours"] = df["hour"].isin([1, 2, 3, 4, 5]).astype(int)

    # 1. 3-day transaction count and personal drift
    df_3d = df[df["days_before"] <= 3.0]
    n_3d = df_3d.groupby(config.ID_COL).size().rename("n_txn_3d")

    # 2. Chiqim vs Kirim amounts
    kirim_amt = df[df["is_kirim"] == 1].groupby(config.ID_COL)["miqdor_indeksi"].sum().rename("amt_sum_kirim")
    chiqim_amt = df[df["is_chiqim"] == 1].groupby(config.ID_COL)["miqdor_indeksi"].sum().rename("amt_sum_chiqim")

    # 3. 7-day cash-out intensity
    df_7d = df[df["days_before"] <= 7.0]
    n_naqd_7d = df_7d[df_7d["is_naqd"] == 1].groupby(config.ID_COL).size().rename("n_naqd_7d")

    # 4. Dark hours share
    dark_share = df.groupby(config.ID_COL)["is_dark_hours"].mean().rename("frac_dark_hours")

    extra = signals[[config.ID_COL]].merge(n_3d, on=config.ID_COL, how="left")
    extra = extra.merge(kirim_amt, on=config.ID_COL, how="left")
    extra = extra.merge(chiqim_amt, on=config.ID_COL, how="left")
    extra = extra.merge(n_naqd_7d, on=config.ID_COL, how="left")
    extra = extra.merge(dark_share, on=config.ID_COL, how="left")
    extra = extra.fillna(0.0)

    # Derived ratios
    extra["vol_ratio_3d_vs_hist"] = (extra["n_txn_3d"] / 3.0) / (X_curr["velocity"] + 0.1)
    extra["ratio_chiqim_to_kirim"] = (extra["amt_sum_chiqim"].abs() + 0.01) / (extra["amt_sum_kirim"].abs() + 0.01)
    extra["cash_intensity_7d"] = extra["n_naqd_7d"] / (X_curr["n_txn_7d"] + 0.1)

    candidates = {
        "Baseline (25 feats, C=0.05)": X_curr,
        "+ vol_ratio_3d_vs_hist": X_curr.assign(vol_ratio_3d_vs_hist=extra["vol_ratio_3d_vs_hist"]),
        "+ ratio_chiqim_to_kirim": X_curr.assign(ratio_chiqim_to_kirim=extra["ratio_chiqim_to_kirim"]),
        "+ cash_intensity_7d": X_curr.assign(cash_intensity_7d=extra["cash_intensity_7d"]),
        "+ frac_dark_hours": X_curr.assign(frac_dark_hours=extra["frac_dark_hours"]),
        "+ ALL 4 new features": X_curr.assign(
            vol_ratio_3d_vs_hist=extra["vol_ratio_3d_vs_hist"],
            ratio_chiqim_to_kirim=extra["ratio_chiqim_to_kirim"],
            cash_intensity_7d=extra["cash_intensity_7d"],
            frac_dark_hours=extra["frac_dark_hours"]
        )
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- Benchmarking Feature Candidates (5-Fold CV) ---")
    for name, cand_X in candidates.items():
        aucs = []
        for tr_idx, va_idx in skf.split(cand_X, y):
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced", random_state=42))
            ])
            pipe.fit(cand_X.iloc[tr_idx], y.iloc[tr_idx])
            p = pipe.predict_proba(cand_X.iloc[va_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va_idx], p))
        print(f"{name:<35} -> AUC: {np.mean(aucs):.5f} +/- {np.std(aucs):.5f}")

if __name__ == "__main__":
    main()
