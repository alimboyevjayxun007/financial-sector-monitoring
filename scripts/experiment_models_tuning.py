"""Systematic Model & Feature tuning script.
Tests:
1. C-parameter regularization scan for LogisticRegression (C in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
2. Impact of amt_mean_1d and amt_mean_7d
3. MLPClassifier tuning (hidden_layer_sizes, alpha)
4. Rank-Average ensemble of LR + MLP
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.data_loading import load_signals, load_transactions
from src.features import build as build_v1

def main():
    print("Loading data...")
    signals = load_signals(config.TRAIN_SIGNALS_PATH)
    txns = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
    y = signals[config.TARGET_COL]

    base_df = build_v1(signals, txns)
    X = base_df[config.FEATURE_COLUMNS].copy()

    # Also compute amt_mean_1d
    df = txns.merge(signals[[config.ID_COL, "signal_sanasi"]], on=config.ID_COL, how="inner")
    df = df[df["tranzaksiya_vaqti"] <= df["signal_sanasi"]].copy()
    df["days_before"] = (df["signal_sanasi"] - df["tranzaksiya_vaqti"]).dt.total_seconds() / 86400

    df_1d = df[df["days_before"] <= 1.0]
    agg_1d = df_1d.groupby(config.ID_COL)["miqdor_indeksi"].mean().rename("amt_mean_1d")
    df_1d_merged = signals[[config.ID_COL]].merge(agg_1d, on=config.ID_COL, how="left").fillna(0.0)
    X_plus_1d = X.copy()
    X_plus_1d["amt_mean_1d"] = df_1d_merged["amt_mean_1d"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- 1. C-Parameter Tuning for Logistic Regression on Base Features ---")
    c_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_c = None
    best_c_auc = 0.0

    for c in c_values:
        aucs = []
        for tr_idx, va_idx in skf.split(X, y):
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=42))
            ])
            pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            p = pipe.predict_proba(X.iloc[va_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va_idx], p))
        mean_auc, std_auc = np.mean(aucs), np.std(aucs)
        print(f"C={c:<6} -> CV ROC-AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
        if mean_auc > best_c_auc:
            best_c_auc = mean_auc
            best_c = c

    print(f"\nBest C on Base Features: C={best_c} with AUC = {best_c_auc:.5f}")

    print("\n--- 2. C-Parameter Tuning on (Base + amt_mean_1d) ---")
    best_c_1d = None
    best_c_1d_auc = 0.0
    for c in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        aucs = []
        for tr_idx, va_idx in skf.split(X_plus_1d, y):
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=42))
            ])
            pipe.fit(X_plus_1d.iloc[tr_idx], y.iloc[tr_idx])
            p = pipe.predict_proba(X_plus_1d.iloc[va_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va_idx], p))
        mean_auc, std_auc = np.mean(aucs), np.std(aucs)
        print(f"C={c:<6} (23 feats) -> CV ROC-AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
        if mean_auc > best_c_1d_auc:
            best_c_1d_auc = mean_auc
            best_c_1d = c

    print(f"\nBest on (Base + amt_mean_1d): C={best_c_1d} with AUC = {best_c_1d_auc:.5f}")

    print("\n--- 3. MLPClassifier Tuning ---")
    mlp_configs = [
        {"hidden_layer_sizes": (16,), "alpha": 0.1},
        {"hidden_layer_sizes": (16,), "alpha": 1.0},
        {"hidden_layer_sizes": (32, 16), "alpha": 1.0},
        {"hidden_layer_sizes": (8, 4), "alpha": 0.5},
    ]
    for cfg in mlp_configs:
        aucs = []
        for tr_idx, va_idx in skf.split(X, y):
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(
                    max_iter=300,
                    random_state=42,
                    early_stopping=True,
                    **cfg
                ))
            ])
            pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            p = pipe.predict_proba(X.iloc[va_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va_idx], p))
        print(f"MLP {cfg} -> CV ROC-AUC: {np.mean(aucs):.5f} +/- {np.std(aucs):.5f}")

    print("\n--- 4. Rank-Average Ensemble (LR + MLP) ---")
    blend_weights = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    oof_lr = np.zeros(len(y))
    oof_mlp = np.zeros(len(y))

    for tr_idx, va_idx in skf.split(X, y):
        m_lr = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced", random_state=42))
        ])
        m_mlp = Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=300, random_state=42, early_stopping=True))
        ])
        m_lr.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        m_mlp.fit(X.iloc[tr_idx], y.iloc[tr_idx])

        oof_lr[va_idx] = m_lr.predict_proba(X.iloc[va_idx])[:, 1]
        oof_mlp[va_idx] = m_mlp.predict_proba(X.iloc[va_idx])[:, 1]

    lr_total_auc = roc_auc_score(y, oof_lr)
    mlp_total_auc = roc_auc_score(y, oof_mlp)
    print(f"OOF Overall LR AUC:  {lr_total_auc:.5f}")
    print(f"OOF Overall MLP AUC: {mlp_total_auc:.5f}")

    rank_lr = rankdata(oof_lr) / len(oof_lr)
    rank_mlp = rankdata(oof_mlp) / len(oof_mlp)

    for w_lr in blend_weights:
        w_mlp = 1.0 - w_lr
        blend = w_lr * rank_lr + w_mlp * rank_mlp
        score = roc_auc_score(y, blend)
        print(f"Blend Weight LR={w_lr:.1f}, MLP={w_mlp:.1f} -> OOF Blend AUC: {score:.5f}")

if __name__ == "__main__":
    main()
