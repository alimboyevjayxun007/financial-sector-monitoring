"""Compute and visualize model interpretation metrics:
Standardized Logistic Regression coefficients, Odds Ratios, Top-10 positive / negative features,
and generate a high-resolution bar plot for presentation and docs.
"""
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src import config, model

def main():
    print("Loading train features...")
    df = pd.read_parquet(config.TRAIN_FEATURES_PATH)
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COL]

    # Fit calibrated model and extract average base estimator coefficients
    fitted_model = model.train(X, y)
    
    # 5-fold ensemble coefficients from CalibratedClassifierCV
    fold_coefs = np.array([
        cc.estimator.named_steps["clf"].coef_[0]
        for cc in fitted_model.calibrated_classifiers_
    ])
    mean_coefs = np.mean(fold_coefs, axis=0)
    std_coefs = np.std(fold_coefs, axis=0)

    # Feature explanations / domain descriptions
    descriptions = {
        "n_txn": "Umumiy tranzaksiyalar soni",
        "amt_mean": "O'rtacha tranzaksiya miqdori",
        "amt_std": "Tranzaksiya miqdorining standart og'ishi",
        "amt_max": "Maksimal tranzaksiya miqdori",
        "amt_sum": "Jami tranzaksiyalar summasi",
        "frac_kirim": "Kirim tranzaksiyalari ulushi",
        "frac_karta": "Karta orqali operatsiyalar ulushi",
        "frac_bank_otkazmasi": "Bank o'tkazmalari ulushi",
        "frac_naqd": "Naqd pul operatsiyalari ulushi",
        "frac_xalqaro": "Xalqaro operatsiyalar ulushi",
        "frac_night": "Tungi operatsiyalar ulushi (00:00-06:00)",
        "frac_weekend": "Dam olish kunlari operatsiyalari ulushi",
        "frac_extreme": "Ekstremal yirik operatsiyalar ulushi",
        "n_txn_1d": "Oxirgi 24 soatdagi tranzaksiyalar soni",
        "n_txn_7d": "Oxirgi 7 kundagi tranzaksiyalar soni",
        "n_txn_30d": "Oxirgi 30 kundagi tranzaksiyalar soni",
        "span_days": "Birinchi va oxirgi tranzaksiya orasidagi kunlar",
        "velocity": "Kunlik o'rtacha tranzaksiyalar tezligi (n_txn / span)",
        "hour_entropy": "Kun soatlari bo'yicha Shannon entropiyasi",
        "hour_maxshare": "Eng faol soatning umumiy tranzaksiyalardagi ulushi",
        "dow_entropy": "Hafta kunlari bo'yicha Shannon entropiyasi",
        "dow_maxshare": "Eng faol hafta kunining ulushi",
        "frac_kirim_1d": "Oxirgi 24 soatdagi kirim operatsiyalari ulushi",
        "ratio_n_1d_to_7d": "24 soatlik faollikning 7 kunlikka nisbati",
        "amt_mean_1d": "Oxirgi 24 soatdagi o'rtacha tranzaksiya miqdori",
    }

    res_df = pd.DataFrame({
        "feature": config.FEATURE_COLUMNS,
        "coef": mean_coefs,
        "coef_std": std_coefs,
        "odds_ratio": np.exp(mean_coefs),
        "abs_coef": np.abs(mean_coefs),
        "description": [descriptions.get(f, "") for f in config.FEATURE_COLUMNS]
    }).sort_values(by="coef", ascending=False).reset_index(drop=True)

    print("\n=== TOP-10 IJOBIY (ESCALATION'GA TORTUVCHI) FEATURE'LAR ===")
    top_pos = res_df.head(10)
    for idx, r in top_pos.iterrows():
        print(f"{r['feature']:<20} | Coef: {r['coef']:+.4f} (std: {r['coef_std']:.4f}) | OR: {r['odds_ratio']:.4f} | {r['description']}")

    print("\n=== TOP-10 SALBIY (DISMISS'GA TORTUVCHI) FEATURE'LAR ===")
    top_neg = res_df.tail(10).iloc[::-1]
    for idx, r in top_neg.iterrows():
        print(f"{r['feature']:<20} | Coef: {r['coef']:+.4f} (std: {r['coef_std']:.4f}) | OR: {r['odds_ratio']:.4f} | {r['description']}")

    # Create visualization
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Sort for horizontal bar plot (negative at top or bottom)
    plot_df = res_df.sort_values(by="coef", ascending=True)
    colors = ["#e74c3c" if c > 0 else "#2980b9" for c in plot_df["coef"]]

    bars = ax.barh(plot_df["feature"], plot_df["coef"], color=colors, alpha=0.85, edgecolor="none", height=0.65)
    ax.axvline(0, color="#2c3e50", linestyle="--", linewidth=1.0, alpha=0.7)

    # Add data labels
    for bar in bars:
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = 0.005 if width >= 0 else -0.005
        ax.text(width + offset, bar.get_y() + bar.get_height() / 2, f"{width:+.3f}",
                va="center", ha=ha, fontsize=8, fontweight="bold", color="#34495e")

    ax.set_title("Standartlashtirilgan Logistic Regression Koeffitsiyentlari\n(AML Signal Eskalatsiyasiga Ta'siri, C=0.05)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Koeffitsiyent (Log-Odds o'zgarishi / 1 std)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Xususiyat (Feature)", fontsize=10, fontweight="bold")
    
    # Custom legend elements
    import matplotlib.patches as mpatches
    pos_patch = mpatches.Patch(color="#e74c3c", label="Eskalatsiyaga tortuvchi (Ijobiy ta'sir)")
    neg_patch = mpatches.Patch(color="#2980b9", label="Dismiss'ga tortuvchi (Salbiy ta'sir)")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", frameon=True)

    plt.tight_layout()
    
    # Ensure directories exist
    fig_dir = config.ROOT / "docs" / "assets"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_fig_path = fig_dir / "feature_importance.png"
    plt.savefig(out_fig_path, bbox_inches="tight")
    
    # Also save to outputs/
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.OUTPUTS_DIR / "feature_importance.png", bbox_inches="tight")
    print(f"\nSaved feature importance figure to:\n - {out_fig_path}\n - {config.OUTPUTS_DIR / 'feature_importance.png'}")

    # Save CSV
    csv_path = config.OUTPUTS_DIR / "feature_importance.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"Saved interpretation table to {csv_path}")

if __name__ == "__main__":
    main()
