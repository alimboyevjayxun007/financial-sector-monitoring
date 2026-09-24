"""
EDA chart generator — WUIT Hackathon, Fintech Track (AML signal escalation).

Bu skript xom ma'lumotlarni (fintech_track_data/fintech_data/*) o'qib,
docs/index.html sahifasida ishlatiladigan barcha PNG grafiklarni
docs/assets/ papkasiga generatsiya qiladi.

Ishga tushirish:
    python3 generate_charts.py

Talab qilinadigan kutubxonalar: pandas, numpy, pyarrow, matplotlib
(loyihaning tizim python3'ida allaqachon o'rnatilgan).

Eslatma: bu skript faqat O'QIYDI — fintech_track_data/ ichidagi hech qanday
faylni o'zgartirmaydi yoki qayta yozmaydi.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sozlamalar
# ---------------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "fintech_track_data" / "fintech_data"
ASSETS_DIR = HERE / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

COLOR_DISMISS = "#3b82f6"   # ko'k — dismiss (0)
COLOR_ESCALATE = "#ef4444"  # qizil — escalate (1)
COLOR_NEUTRAL = "#6366f1"   # binafsha-ko'k — neytral grafiklar
COLOR_ACCENT = "#10b981"    # yashil — ikkinchi urg'u rangi

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)


def savefig(fig, name: str) -> None:
    out_path = ASSETS_DIR / name
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    size_kb = out_path.stat().st_size / 1024
    print(f"  saqlandi: {out_path.relative_to(REPO_ROOT)}  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# 1. Ma'lumotlarni yuklash
# ---------------------------------------------------------------------------

print("Ma'lumotlar yuklanmoqda...")

train_signals = pd.read_csv(DATA_DIR / "train_signals.csv")
train_signals["signal_sanasi"] = pd.to_datetime(train_signals["signal_sanasi"])

test_signals = pd.read_csv(DATA_DIR / "test_signals.csv")
test_signals["signal_sanasi"] = pd.to_datetime(test_signals["signal_sanasi"])

train_txn = pd.read_parquet(DATA_DIR / "train_transactions.parquet")
test_txn = pd.read_parquet(DATA_DIR / "test_transactions.parquet")

print(
    f"  train_signals: {len(train_signals):,} qator | "
    f"test_signals: {len(test_signals):,} qator"
)
print(
    f"  train_transactions: {len(train_txn):,} qator | "
    f"test_transactions: {len(test_txn):,} qator"
)

# Barcha tranzaksiyalar (train+test) — dataset darajasidagi umumiy ko'rinish uchun
all_txn = pd.concat([train_txn, test_txn], ignore_index=True)

# ---------------------------------------------------------------------------
# 2. Signal darajasidagi agregat xususiyatlarni hisoblash (faqat train uchun,
#    leakage'ga qarshi qoida bilan: faqat tranzaksiya_vaqti <= signal_sanasi)
# ---------------------------------------------------------------------------

print("Signal darajasidagi agregat xususiyatlar hisoblanmoqda (leakage filtri bilan)...")


def build_features(signals: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    merged = txn.merge(
        signals[["signal_id", "signal_sanasi"]], on="signal_id", how="inner"
    )
    # Leakage'ga qarshi: faqat signal sanasidan oldingi/tengdagi tranzaksiyalar
    merged = merged[merged["tranzaksiya_vaqti"] <= merged["signal_sanasi"]].copy()

    merged["is_kirim"] = (merged["kirim_chiqim"] == "kirim").astype(int)
    merged["is_naqd"] = (merged["tranzaksiya_turi"] == "naqd").astype(int)
    merged["is_karta"] = (merged["tranzaksiya_turi"] == "karta").astype(int)
    merged["is_extreme"] = (merged["miqdor_indeksi"].abs() > 2).astype(int)
    merged["days_before_signal"] = (
        merged["signal_sanasi"] - merged["tranzaksiya_vaqti"]
    ).dt.total_seconds() / 86400.0

    grp = merged.groupby("signal_id")
    feats = grp.agg(
        n_txn=("miqdor_indeksi", "size"),
        amt_mean=("miqdor_indeksi", "mean"),
        amt_std=("miqdor_indeksi", "std"),
        amt_max=("miqdor_indeksi", "max"),
        amt_sum=("miqdor_indeksi", "sum"),
        frac_kirim=("is_kirim", "mean"),
        frac_naqd=("is_naqd", "mean"),
        frac_karta=("is_karta", "mean"),
        frac_extreme=("is_extreme", "mean"),
    )
    feats["n_txn_1d"] = grp.apply(
        lambda g: (g["days_before_signal"] <= 1).sum(), include_groups=False
    )
    feats["n_txn_7d"] = grp.apply(
        lambda g: (g["days_before_signal"] <= 7).sum(), include_groups=False
    )
    feats["n_txn_30d"] = grp.apply(
        lambda g: (g["days_before_signal"] <= 30).sum(), include_groups=False
    )
    feats = feats.reset_index()
    feats["amt_std"] = feats["amt_std"].fillna(0.0)
    out = feats.merge(signals[["signal_id", "eskalatsiya"]], on="signal_id", how="left")
    return out


train_feats = build_features(train_signals, train_txn)
print(f"  train feature jadvali: {train_feats.shape}")

# ---------------------------------------------------------------------------
# 3. Chart 1 — Target sinf taqsimoti
# ---------------------------------------------------------------------------

print("Grafik 1/8: target taqsimoti...")

target_counts = train_signals["eskalatsiya"].value_counts().sort_index()
labels = ["Dismiss (0)", "Escalate (1)"]
counts = [target_counts.get(0, 0), target_counts.get(1, 0)]
pct = [c / sum(counts) * 100 for c in counts]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(labels, counts, color=[COLOR_DISMISS, COLOR_ESCALATE], width=0.55)
for bar, c, p in zip(bars, counts, pct):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(counts) * 0.02,
        f"{c:,}\n({p:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
ax.set_ylabel("Signallar soni")
ax.set_title("Target taqsimoti: dismiss vs escalate (train, n=14 000)", fontsize=12)
ax.set_ylim(0, max(counts) * 1.2)
savefig(fig, "01_target_distribution.png")

# ---------------------------------------------------------------------------
# 4. Chart 2 — Vaqt bo'yicha tranzaksiya hajmi (haftalik)
# ---------------------------------------------------------------------------

print("Grafik 2/8: vaqt bo'yicha tranzaksiya hajmi...")

weekly = (
    all_txn.set_index("tranzaksiya_vaqti")
    .resample("W")
    .size()
)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(weekly.index, weekly.values, color=COLOR_NEUTRAL, linewidth=1.4)
ax.fill_between(weekly.index, weekly.values, color=COLOR_NEUTRAL, alpha=0.15)
ax.set_ylabel("Tranzaksiyalar soni (haftalik)")
ax.set_title("Tranzaksiya hajmi vaqt bo'yicha (train + test, haftalik)")
ax.set_xlabel("Sana")
fig.autofmt_xdate()
savefig(fig, "02_transactions_over_time.png")

# ---------------------------------------------------------------------------
# 5. Chart 3 — kirim vs chiqim
# ---------------------------------------------------------------------------

print("Grafik 3/8: kirim vs chiqim...")

dir_counts = all_txn["kirim_chiqim"].value_counts()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
wedges, texts, autotexts = ax.pie(
    dir_counts.values,
    labels=[f"{i} ({v/dir_counts.sum()*100:.1f}%)" for i, v in dir_counts.items()],
    colors=[COLOR_ACCENT, "#f59e0b"],
    autopct=lambda p: f"{p:.0f}%" if False else "",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
ax.set_title(
    "Yo'nalish taqsimoti: kirim vs chiqim\n(barcha tranzaksiyalar)", fontsize=12
)
savefig(fig, "03_kirim_chiqim.png")

# ---------------------------------------------------------------------------
# 6. Chart 4 — tranzaksiya_turi taqsimoti
# ---------------------------------------------------------------------------

print("Grafik 4/8: tranzaksiya turi taqsimoti...")

type_counts = all_txn["tranzaksiya_turi"].value_counts()
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.barh(type_counts.index[::-1], type_counts.values[::-1], color=COLOR_NEUTRAL)
total = type_counts.sum()
for bar, v in zip(bars, type_counts.values[::-1]):
    ax.text(
        bar.get_width() + total * 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{v:,} ({v/total*100:.1f}%)",
        va="center",
        fontsize=10,
    )
ax.set_xlabel("Tranzaksiyalar soni")
ax.set_title("Tranzaksiya turi bo'yicha taqsimot")
ax.set_xlim(0, type_counts.max() * 1.25)
savefig(fig, "04_tranzaksiya_turi.png")

# ---------------------------------------------------------------------------
# 7. Chart 5 — miqdor_indeksi (amt_mean per signal), escalate vs dismiss
# ---------------------------------------------------------------------------

print("Grafik 5/8: amt_mean taqsimoti (escalate vs dismiss)...")

dismiss_amt = train_feats.loc[train_feats["eskalatsiya"] == 0, "amt_mean"]
escalate_amt = train_feats.loc[train_feats["eskalatsiya"] == 1, "amt_mean"]

fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(
    train_feats["amt_mean"].quantile(0.01),
    train_feats["amt_mean"].quantile(0.99),
    40,
)
ax.hist(
    dismiss_amt, bins=bins, alpha=0.55, label="Dismiss (0)", color=COLOR_DISMISS,
    density=True,
)
ax.hist(
    escalate_amt, bins=bins, alpha=0.55, label="Escalate (1)", color=COLOR_ESCALATE,
    density=True,
)
ax.set_xlabel("Signal bo'yicha o'rtacha miqdor_indeksi (amt_mean)")
ax.set_ylabel("Zichlik (density)")
ax.set_title("Signal bo'yicha o'rtacha tranzaksiya summasi: escalate vs dismiss")
ax.legend()
savefig(fig, "05_amt_mean_distribution.png")

# ---------------------------------------------------------------------------
# 8. Chart 6 — signal boshiga tranzaksiyalar soni taqsimoti
# ---------------------------------------------------------------------------

print("Grafik 6/8: signal boshiga tranzaksiyalar soni...")

fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(0, train_feats["n_txn"].quantile(0.99), 40)
ax.hist(
    train_feats.loc[train_feats["eskalatsiya"] == 0, "n_txn"],
    bins=bins,
    alpha=0.55,
    label="Dismiss (0)",
    color=COLOR_DISMISS,
    density=True,
)
ax.hist(
    train_feats.loc[train_feats["eskalatsiya"] == 1, "n_txn"],
    bins=bins,
    alpha=0.55,
    label="Escalate (1)",
    color=COLOR_ESCALATE,
    density=True,
)
ax.axvline(
    train_feats["n_txn"].median(),
    color="black",
    linestyle=":",
    linewidth=1.2,
    label=f"Median = {train_feats['n_txn'].median():.0f}",
)
ax.set_xlabel("Signal boshiga tranzaksiyalar soni (n_txn)")
ax.set_ylabel("Zichlik (density)")
ax.set_title("Signal boshiga tranzaksiyalar soni taqsimoti")
ax.legend()
savefig(fig, "06_n_txn_distribution.png")

# ---------------------------------------------------------------------------
# 9. Chart 7 — signal oldidan so'nggi 1/7/30 kunlik faollik, target bo'yicha
# ---------------------------------------------------------------------------

print("Grafik 7/8: signal oldidan so'nggi faollik...")

recency_cols = ["n_txn_1d", "n_txn_7d", "n_txn_30d"]
recency_means = train_feats.groupby("eskalatsiya")[recency_cols].mean()

x = np.arange(len(recency_cols))
width = 0.35

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.bar(
    x - width / 2,
    recency_means.loc[0].values,
    width,
    label="Dismiss (0)",
    color=COLOR_DISMISS,
)
ax.bar(
    x + width / 2,
    recency_means.loc[1].values,
    width,
    label="Escalate (1)",
    color=COLOR_ESCALATE,
)
for i, col in enumerate(recency_cols):
    ax.text(
        i - width / 2, recency_means.loc[0, col] + 0.6,
        f"{recency_means.loc[0, col]:.1f}", ha="center", fontsize=9,
    )
    ax.text(
        i + width / 2, recency_means.loc[1, col] + 0.6,
        f"{recency_means.loc[1, col]:.1f}", ha="center", fontsize=9,
    )
ax.set_xticks(x)
ax.set_xticklabels(["so'nggi 1 kun", "so'nggi 7 kun", "so'nggi 30 kun"])
ax.set_ylabel("O'rtacha tranzaksiyalar soni")
ax.set_title("Signal sanasidan oldingi faollik: escalate vs dismiss")
ax.legend()
savefig(fig, "07_recent_activity.png")

# ---------------------------------------------------------------------------
# 10. Chart 8 — agregat xususiyatlarning target bilan korrelyatsiyasi
# ---------------------------------------------------------------------------

print("Grafik 8/8: xususiyatlar korrelyatsiyasi...")

corr_cols = [
    "n_txn", "amt_mean", "amt_std", "amt_max", "amt_sum",
    "frac_kirim", "frac_naqd", "frac_karta", "frac_extreme",
    "n_txn_1d", "n_txn_7d", "n_txn_30d",
]
corrs = train_feats[corr_cols + ["eskalatsiya"]].corr()["eskalatsiya"].drop("eskalatsiya")
corrs = corrs.sort_values()

fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors = [COLOR_ESCALATE if v > 0 else COLOR_DISMISS for v in corrs.values]
ax.barh(corrs.index, corrs.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson korrelyatsiya target (eskalatsiya) bilan")
ax.set_title("Agregat xususiyatlarning target bilan korrelyatsiyasi")
x_pad = (corrs.max() - corrs.min()) * 0.12
ax.set_xlim(corrs.min() - x_pad * 2.2, corrs.max() + x_pad * 2.2)
for i, (name, v) in enumerate(corrs.items()):
    ax.text(
        v + (x_pad * 0.3 if v >= 0 else -x_pad * 0.3),
        i,
        f"{v:+.3f}",
        va="center",
        ha="left" if v >= 0 else "right",
        fontsize=9,
    )
savefig(fig, "08_feature_correlations.png")

print("\nBarcha grafiklar tayyor:", ASSETS_DIR)
print("\nQo'shimcha tekshirilgan raqamlar (sahifada ishlatish uchun):")
print(f"  train signallar: {len(train_signals):,}, test signallar: {len(test_signals):,}")
print(f"  escalate ulushi: {counts[1] / sum(counts) * 100:.1f}%")
print(f"  n_txn per signal: min={train_feats['n_txn'].min()}, "
      f"median={train_feats['n_txn'].median():.0f}, max={train_feats['n_txn'].max()}")
print("  amt_mean corr:", round(corrs.get("amt_mean", float('nan')), 4))
print("  amt_max corr:", round(corrs.get("amt_max", float('nan')), 4))
print("  amt_std corr:", round(corrs.get("amt_std", float('nan')), 4))
print("  frac_kirim corr:", round(corrs.get("frac_kirim", float('nan')), 4))
print("  n_txn corr:", round(corrs.get("n_txn", float('nan')), 4))
print("  frac_naqd corr:", round(corrs.get("frac_naqd", float('nan')), 4))
print("  n_txn_7d corr:", round(corrs.get("n_txn_7d", float('nan')), 4))
print("  n_txn_1d corr:", round(corrs.get("n_txn_1d", float('nan')), 4))
print("  frac_extreme corr:", round(corrs.get("frac_extreme", float('nan')), 4))
print(
    "  n_txn mean (dismiss vs escalate):",
    train_feats.groupby("eskalatsiya")["n_txn"].mean().round(1).to_dict(),
)
print(
    "  n_txn_7d mean (dismiss vs escalate):",
    train_feats.groupby("eskalatsiya")["n_txn_7d"].mean().round(1).to_dict(),
)
print(
    "  n_txn_1d mean (dismiss vs escalate):",
    train_feats.groupby("eskalatsiya")["n_txn_1d"].mean().round(1).to_dict(),
)
