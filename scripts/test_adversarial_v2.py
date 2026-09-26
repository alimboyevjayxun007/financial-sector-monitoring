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

def build_25(signals, txns):
    df_base = build_v1(signals, txns)
    X = df_base[config.FEATURE_COLUMNS].copy()

    df = txns.merge(signals[[config.ID_COL, "signal_sanasi"]], on=config.ID_COL, how="inner")
    df = df[df["tranzaksiya_vaqti"] <= df["signal_sanasi"]].copy()
    df["days_before"] = (df["signal_sanasi"] - df["tranzaksiya_vaqti"]).dt.total_seconds() / 86400
    df["is_kirim"] = (df["kirim_chiqim"] == "kirim").astype(int)

    df_1d = df[df["days_before"] <= 1.0]
    agg_1d = df_1d.groupby(config.ID_COL).agg(
        amt_mean_1d=("miqdor_indeksi", "mean"),
        frac_kirim_1d=("is_kirim", "mean"),
    ).reset_index()

    cand = signals[[config.ID_COL]].merge(agg_1d, on=config.ID_COL, how="left").fillna(0.0)
    cand["ratio_n_1d_to_7d"] = (X["n_txn_1d"]) / ((X["n_txn_7d"] / 7.0) + 0.1)

    X["frac_kirim_1d"] = cand["frac_kirim_1d"]
    X["ratio_n_1d_to_7d"] = cand["ratio_n_1d_to_7d"]
    X["amt_mean_1d"] = cand["amt_mean_1d"]
    return X

def main():
    print("Loading data...")
    train_signals = load_signals(config.TRAIN_SIGNALS_PATH)
    train_txns = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
    test_signals = load_signals(config.TEST_SIGNALS_PATH)
    test_txns = load_transactions(config.TEST_TRANSACTIONS_PATH)

    print("Building 25-feature sets...")
    X_train = build_25(train_signals, train_txns)
    X_test = build_25(test_signals, test_txns)

    X_all = pd.concat([X_train, X_test], ignore_index=True)
    y_domain = np.array([0] * len(X_train) + [1] * len(X_test))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, va_idx in skf.split(X_all, y_domain):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])
        pipe.fit(X_all.iloc[tr_idx], y_domain[tr_idx])
        p = pipe.predict_proba(X_all.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(y_domain[va_idx], p))

    mean_auc = np.mean(aucs)
    print(f"Adversarial Validation AUC on 25 features: {mean_auc:.5f} +/- {np.std(aucs):.5f}")
    if abs(mean_auc - 0.5) < 0.02:
        print("PERFECT: No detectable covariate shift between train and test!")
    else:
        print("WARNING: Covariate shift detected!")

if __name__ == "__main__":
    main()
