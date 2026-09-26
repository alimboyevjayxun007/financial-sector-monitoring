import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.data_loading import load_signals, load_transactions
from src.features import build as build_v1

def nadeau_bengio_pvalue(diffs, n_train_rel=0.8, n_test_rel=0.2):
    n = len(diffs)
    mean_d = np.mean(diffs)
    var_d = np.var(diffs, ddof=1)
    corrected_var = var_d * (1.0 / n + n_test_rel / n_train_rel)
    if corrected_var <= 0:
        return 0.0, 1.0
    t_stat = mean_d / np.sqrt(corrected_var)
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    return t_stat, p_val

def main():
    print("Loading data...")
    signals = load_signals(config.TRAIN_SIGNALS_PATH)
    txns = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
    y = signals[config.TARGET_COL]

    base_df = build_v1(signals, txns)
    X_base = base_df[config.FEATURE_COLUMNS].copy()

    df = txns.merge(signals[[config.ID_COL, "signal_sanasi"]], on=config.ID_COL, how="inner")
    df = df[df["tranzaksiya_vaqti"] <= df["signal_sanasi"]].copy()
    df["days_before"] = (df["signal_sanasi"] - df["tranzaksiya_vaqti"]).dt.total_seconds() / 86400
    df["is_kirim"] = (df["kirim_chiqim"] == "kirim").astype(int)

    df_1d = df[df["days_before"] <= 1.0]
    agg_1d = df_1d.groupby(config.ID_COL).agg(
        amt_mean_1d=("miqdor_indeksi", "mean"),
        frac_kirim_1d=("is_kirim", "mean"),
    ).reset_index()

    candidates = signals[[config.ID_COL]].merge(agg_1d, on=config.ID_COL, how="left").fillna(0.0)
    candidates["ratio_n_1d_to_7d"] = (X_base["n_txn_1d"]) / ((X_base["n_txn_7d"] / 7.0) + 0.1)

    combos = {
        "Base (22 cols)": X_base,
        "Base + frac_kirim_1d": X_base.assign(frac_kirim_1d=candidates["frac_kirim_1d"]),
        "Base + frac_kirim_1d + ratio_n_1d_to_7d": X_base.assign(
            frac_kirim_1d=candidates["frac_kirim_1d"],
            ratio_n_1d_to_7d=candidates["ratio_n_1d_to_7d"]
        ),
        "Base + frac_kirim_1d + amt_mean_1d": X_base.assign(
            frac_kirim_1d=candidates["frac_kirim_1d"],
            amt_mean_1d=candidates["amt_mean_1d"]
        ),
        "Base + 3 Recency (frac_kirim_1d, ratio_n_1d_to_7d, amt_mean_1d)": X_base.assign(
            frac_kirim_1d=candidates["frac_kirim_1d"],
            ratio_n_1d_to_7d=candidates["ratio_n_1d_to_7d"],
            amt_mean_1d=candidates["amt_mean_1d"]
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- 5-Fold Cross Validation ---")
    for name, X_cand in combos.items():
        aucs = []
        for tr_idx, va_idx in skf.split(X_cand, y):
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42))
            ])
            pipe.fit(X_cand.iloc[tr_idx], y.iloc[tr_idx])
            p = pipe.predict_proba(X_cand.iloc[va_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va_idx], p))
        print(f"{name:<55} -> AUC: {np.mean(aucs):.5f} +/- {np.std(aucs):.5f}")

    print("\n--- 5x5 Repeated Stratified CV Comparison (Base vs Top Combo) ---")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    base_scores = []
    combo_scores = []
    X_top = combos["Base + frac_kirim_1d + amt_mean_1d"]

    for tr_idx, va_idx in rskf.split(X_base, y):
        pipe_base = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42))
        ])
        pipe_base.fit(X_base.iloc[tr_idx], y.iloc[tr_idx])
        p_base = pipe_base.predict_proba(X_base.iloc[va_idx])[:, 1]
        base_scores.append(roc_auc_score(y.iloc[va_idx], p_base))

        pipe_combo = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42))
        ])
        pipe_combo.fit(X_top.iloc[tr_idx], y.iloc[tr_idx])
        p_combo = pipe_combo.predict_proba(X_top.iloc[va_idx])[:, 1]
        combo_scores.append(roc_auc_score(y.iloc[va_idx], p_combo))

    diffs = np.array(combo_scores) - np.array(base_scores)
    t_stat, p_val = nadeau_bengio_pvalue(diffs)
    print(f"Base (22 cols) Repeated Mean AUC: {np.mean(base_scores):.5f} +/- {np.std(base_scores):.5f}")
    print(f"Top Combo (24 cols) Repeated Mean AUC: {np.mean(combo_scores):.5f} +/- {np.std(combo_scores):.5f}")
    print(f"Mean Difference: +{np.mean(diffs):.5f}")
    print(f"Nadeau-Bengio corrected t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")

if __name__ == "__main__":
    main()
