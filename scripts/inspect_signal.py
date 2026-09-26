import argparse
import sys
from pathlib import Path
from typing import Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from src import config

OPTIMAL_THRESHOLD = 0.1560

DESCRIPTIONS = {
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
    "span_days": "Faollik davri (kunlar)",
    "velocity": "Tranzaksiyalar tezligi (n_txn / span)",
    "hour_entropy": "Kun soatlari bo'yicha Shannon entropiyasi",
    "hour_maxshare": "Eng faol soatning umumiy ulushi",
    "dow_entropy": "Hafta kunlari bo'yicha Shannon entropiyasi",
    "dow_maxshare": "Eng faol hafta kunining ulushi",
    "frac_kirim_1d": "Oxirgi 24 soatdagi kirim ulushi",
    "ratio_n_1d_to_7d": "24 soatlik faollikning 7 kunlikka nisbati",
    "amt_mean_1d": "Oxirgi 24 soatdagi o'rtacha tranzaksiya miqdori",
}


def load_resources():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"{config.MODEL_PATH} not found. Run `python -m src.train` first.")
    model = joblib.load(config.MODEL_PATH)

    test_df = pd.read_parquet(config.TEST_FEATURES_PATH) if config.TEST_FEATURES_PATH.exists() else None
    train_df = pd.read_parquet(config.TRAIN_FEATURES_PATH) if config.TRAIN_FEATURES_PATH.exists() else None

    base_estimators = [cc.estimator for cc in model.calibrated_classifiers_]
    mean_coef = np.mean([be.named_steps["clf"].coef_[0] for be in base_estimators], axis=0)
    mean_scale = np.mean([be.named_steps["scale"].scale_ for be in base_estimators], axis=0)
    mean_center = np.mean([be.named_steps["scale"].mean_ for be in base_estimators], axis=0)

    return model, train_df, test_df, mean_coef, mean_center, mean_scale


def inspect_single_signal(
    signal_id: str,
    df: pd.DataFrame,
    model,
    mean_coef: np.ndarray,
    mean_center: np.ndarray,
    mean_scale: np.ndarray,
):
    row_match = df[df[config.ID_COL] == signal_id]
    if row_match.empty:
        print(f"Error: signal_id '{signal_id}' topilmadi.")
        return

    row = row_match.iloc[0]
    X_raw = row[config.FEATURE_COLUMNS].to_numpy(dtype=float)
    proba = float(model.predict_proba(row_match[config.FEATURE_COLUMNS])[0, 1])

    X_std = (X_raw - mean_center) / mean_scale
    contributions = X_std * mean_coef

    contrib_df = pd.DataFrame({
        "feature": config.FEATURE_COLUMNS,
        "raw_value": X_raw,
        "standardized": X_std,
        "coef": mean_coef,
        "contribution": contributions,
        "abs_contrib": np.abs(contributions),
        "desc": [DESCRIPTIONS.get(f, "") for f in config.FEATURE_COLUMNS],
    })

    print("=" * 78)
    print(f"  🔍 SIGNAL TAHLILI: {signal_id}")
    print("=" * 78)
    print(f"  • Eskalatsiya ehtimolligi (Calibrated): {proba * 100:.2f}%")
    print(f"  • Bazaviy stavka:                      17.20%")
    print(f"  • Optimal qaror threshold'i:           {OPTIMAL_THRESHOLD * 100:.2f}%")

    if proba >= OPTIMAL_THRESHOLD:
        print("  • TAVSIYA ETILADIGAN QAROR:            🚨 ESCALATE (Xavfli - mutaxassis tekshiruvi)")
    else:
        print("  • TAVSIYA ETILADIGAN QAROR:            ✅ DISMISS (Past xavf - soxta signal)")
    print("-" * 78)

    print("\n  🚩 ENG KUCHLI XAVF DRAYVERLARI (Eskalatsiyaga tortuvchi omillar):")
    print(f"  {'Xususiyat':<20} | {'Xom qiymat':<12} | {'Ta\'sir (log-odds)':<18} | {'Mazmuni'}")
    print("  " + "-" * 74)
    top_pos = contrib_df[contrib_df["contribution"] > 0].sort_values(by="contribution", ascending=False).head(5)
    for _, r in top_pos.iterrows():
        bar = "█" * min(15, max(1, int(r["contribution"] * 12)))
        print(f"  {r['feature']:<20} | {r['raw_value']:<12.3f} | +{r['contribution']:<6.3f} {bar:<8} | {r['desc']}")

    print("\n  🛡️ ENG KUCHLI YUMSHATUVCHI OMILLAR (Dismiss'ga tortuvchi omillar):")
    print(f"  {'Xususiyat':<20} | {'Xom qiymat':<12} | {'Ta\'sir (log-odds)':<18} | {'Mazmuni'}")
    print("  " + "-" * 74)
    top_neg = contrib_df[contrib_df["contribution"] < 0].sort_values(by="contribution", ascending=True).head(5)
    for _, r in top_neg.iterrows():
        bar = "░" * min(15, max(1, int(abs(r["contribution"]) * 12)))
        print(f"  {r['feature']:<20} | {r['raw_value']:<12.3f} | {r['contribution']:<7.3f} {bar:<8} | {r['desc']}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AML Signal Inspector & Explainability Tool")
    parser.add_argument("--signal-id", type=str, help="Tekshiriladigan signal_id (masalan, SG_0001)")
    parser.add_argument("--top-risks", type=int, default=0, help="Eng yuqori xavfli N ta signalni ko'rsatish")
    args = parser.parse_args()

    model, train_df, test_df, mean_coef, mean_center, mean_scale = load_resources()

    active_df = test_df if test_df is not None else train_df
    if active_df is None:
        print("Feature fayllari topilmadi. Avval `python -m src.features` ni ishga tushiring.")
        return

    if args.signal_id:
        df_target = active_df
        if args.signal_id not in df_target[config.ID_COL].values and train_df is not None:
            df_target = train_df
        inspect_single_signal(args.signal_id, df_target, model, mean_coef, mean_center, mean_scale)
    elif args.top_risks > 0:
        print(f"\n⚡ Test to'plamidagi eng yuqori riskli {args.top_risks} ta signal tahlili:\n")
        preds = model.predict_proba(active_df[config.FEATURE_COLUMNS])[:, 1]
        active_df_copy = active_df.copy()
        active_df_copy["proba"] = preds
        top_ids = active_df_copy.sort_values(by="proba", ascending=False).head(args.top_risks)[config.ID_COL].tolist()
        for sid in top_ids:
            inspect_single_signal(sid, active_df, model, mean_coef, mean_center, mean_scale)
    else:
        print("\n⚡ Signal Inspector Demo (misol tariqasida eng yuqori va eng past riskli signallar):\n")
        preds = model.predict_proba(active_df[config.FEATURE_COLUMNS])[:, 1]
        active_df_copy = active_df.copy()
        active_df_copy["proba"] = preds
        highest_id = active_df_copy.sort_values(by="proba", ascending=False).iloc[0][config.ID_COL]
        lowest_id = active_df_copy.sort_values(by="proba", ascending=True).iloc[0][config.ID_COL]
        inspect_single_signal(highest_id, active_df, model, mean_coef, mean_center, mean_scale)
        inspect_single_signal(lowest_id, active_df, model, mean_coef, mean_center, mean_scale)


if __name__ == "__main__":
    main()
