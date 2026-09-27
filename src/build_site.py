"""Render the EDA website as static HTML with pre-rendered charts."""
import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import pandas as pd

from src import config
from src.data import load_signals, load_transactions
from src.features import prepare_transactions
from src.site_data import compute

PALETTE = {"Dismissed": "#4a8f79", "Escalated": "#d36a52"}


def _figure_to_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight", transparent=True)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _charts(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    charts: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    target = tables["target"]
    ax.bar(target["outcome"], target["alerts"], color=[PALETTE[o] for o in target["outcome"]])
    ax.set_ylabel("Alerts")
    ax.set_title("Target distribution")
    charts["target"] = _figure_to_data_uri(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    weekly = tables["weekly_volume"]
    ax.plot(weekly["tranzaksiya_vaqti"], weekly["transactions"], color="#3f6f8f")
    ax.set_ylabel("Transactions")
    ax.set_title("Weekly transaction volume")
    charts["weekly"] = _figure_to_data_uri(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_alpha(0.0)
    axes[0].patch.set_alpha(0.0)
    axes[1].patch.set_alpha(0.0)
    direction = tables["direction"]
    axes[0].bar(direction["direction"], direction["transactions"], color="#3f6f8f")
    axes[0].set_title("Direction")
    types = tables["types"]
    axes[1].bar(types["type"], types["transactions"], color="#7b6f9f")
    axes[1].set_title("Transaction type")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    charts["direction_types"] = _figure_to_data_uri(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    hist = tables["days_before_hist"]
    for outcome, group in hist.groupby("eskalatsiya"):
        share = group["transactions"] / group["transactions"].sum()
        ax.plot(group["bucket"], share, label=outcome, color=PALETTE[outcome])
    ax.set_xlabel("Days before the alert")
    ax.set_ylabel("Share of transactions")
    ax.set_title("Activity leading up to the alert")
    ax.legend()
    charts["days_before"] = _figure_to_data_uri(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    by_type = tables["type_by_outcome"].pivot(
        index="tranzaksiya_turi", columns="eskalatsiya", values="transactions"
    )
    share = by_type.div(by_type.sum(axis=0), axis=1)
    share.plot.bar(ax=ax, color=[PALETTE[c] for c in share.columns])
    ax.set_ylabel("Share within outcome")
    ax.set_title("Transaction type mix by outcome")
    ax.tick_params(axis="x", rotation=20)
    charts["type_outcome"] = _figure_to_data_uri(fig)

    return charts


def build(output_dir: Path | None = None) -> Path:
    output_dir = config.DOCS if output_dir is None else output_dir
    signals = load_signals(config.TRAIN_SIGNALS)
    tx = prepare_transactions(load_transactions(config.TRAIN_TX), signals)
    tables = compute(signals, tx)
    charts = _charts(tables)

    summary_path = config.EXPERIMENTS / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    ablation_rows = "".join(
        f'<tr class="hover:bg-white/[0.02] transition-colors">'
        f'<td class="py-1.5 pl-4 text-slate-200">{row["family"]}</td>'
        f'<td class="py-1.5 text-center font-mono text-slate-300">{row["n_columns"]}</td>'
        f'<td class="py-1.5 text-center font-mono text-slate-300">{row["mean"]:.4f}</td>'
        f'<td class="py-1.5 pr-4 text-center">'
        f'<span class="inline-block min-w-[62px] text-center px-2.5 py-0.5 rounded-full text-[11px] font-medium '
        + ("bg-[#34d399] text-[#042f1a] shadow-[0_0_10px_rgba(52,211,153,0.35)]" if row["accepted"] else "bg-[#ef4444] text-white shadow-[0_0_8px_rgba(239,68,68,0.35)]")
        + f'">{"kept" if row["accepted"] else "rejected"}</span></td></tr>'
        for row in summary.get("selection_history", [])
    )

    html = _TEMPLATE.format(
        rate=f"{signals['eskalatsiya'].mean():.1%}",
        alerts=f"{len(signals):,}",
        transactions=f"{len(tx):,}",
        columns=summary.get("n_columns", "—"),
        auc=f"{summary.get('ensemble_cv_mean', float('nan')):.4f}",
        auc_std=f"{summary.get('ensemble_cv_std', float('nan')):.4f}",
        best_single=summary.get("best_single_model", "—"),
        best_single_auc=f"{summary.get('best_single_cv_mean', float('nan')):.4f}",
        gain=f"{summary.get('ensemble_gain_over_best_single', float('nan')):+.4f}",
        chance_families=len({
            r["family"] for r in summary.get("selection_history", [])
            if r["score"] <= 0.5
        }),
        adversarial=(
            f"{summary['adversarial_auc']:.4f}"
            if summary.get("adversarial_auc") is not None else "not yet measured"
        ),
        drift_verdict=(
            "indistinguishable"
            if (summary.get("adversarial_auc") or 1.0) < 0.55
            else "separable — treat the estimate with caution"
        ),
        rejected_families=len({
            r["family"] for r in summary.get("selection_history", [])
        }) - len(summary.get("families", [])),
        ablation_rows=ablation_rows or "<tr><td colspan='5'>Run the pipeline first.</td></tr>",
        **charts,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "index.html"
    target.write_text(html, encoding="utf-8")
    (output_dir / ".nojekyll").write_text("")
    return target


_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>AML Alert Prioritization - EDA</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<script>
    tailwind.config = {{
      theme: {{
        extend: {{
          colors: {{
            brandDark: '#080c11',
            cardBg: 'rgba(13, 19, 28, 0.72)',
            calloutBorder: '#4ade80',
            badgeGreen: '#22c55e',
            badgeRed: '#ef4444'
          }},
          fontFamily: {{
            sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
            hud: ['Rajdhani', 'sans-serif']
          }}
        }}
      }}
    }}
</script>
<style>
    body {{ background-color: #060b11; color: #e2e8f0; font-family: 'Inter', sans-serif; overflow-x: hidden; margin: 0; }}
    .font-hud {{ font-family: 'Rajdhani', sans-serif; }}
    .glow-cyan {{ filter: drop-shadow(0 0 10px rgba(6, 182, 212, 0.6)) drop-shadow(0 0 20px rgba(16, 185, 129, 0.3)); }}
    .glow-cyan-lg {{ filter: drop-shadow(0 0 18px rgba(34, 211, 238, 0.75)) drop-shadow(0 0 35px rgba(52, 211, 153, 0.45)); }}
    .glow-text {{ text-shadow: 0 0 12px rgba(74, 222, 128, 0.65), 0 0 24px rgba(34, 211, 238, 0.4); }}
    @keyframes spin-slow {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
    @keyframes spin-reverse {{ 0% {{ transform: rotate(360deg); }} 100% {{ transform: rotate(0deg); }} }}
    @keyframes spin-fast {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
    @keyframes pulse-subtle {{ 0%, 100% {{ opacity: 0.8; transform: scale(1); }} 50% {{ opacity: 1; transform: scale(1.02); }} }}
    .animate-spin-slow {{ animation: spin-slow 24s linear infinite; }}
    .animate-spin-reverse {{ animation: spin-reverse 18s linear infinite; }}
    .animate-spin-fast {{ animation: spin-fast 9s linear infinite; }}
    .animate-pulse-glow {{ animation: pulse-subtle 4s ease-in-out infinite; }}
    .glass-card {{ background: linear-gradient(145deg, rgba(19, 26, 43, 0.85) 0%, rgba(13, 18, 30, 0.85) 100%); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5); }}
    .prism-background {{ position: absolute; width: 540px; height: 520px; left: 50%; top: 55%; transform: translate(-50%, -46%) rotate(12deg); background: radial-gradient(ellipse at 35% 35%, rgba(255, 255, 255, 0.42) 0%, rgba(255, 215, 0, 0.16) 24%, rgba(255, 80, 80, 0.18) 42%, rgba(0, 180, 255, 0.22) 65%, transparent 80%), conic-gradient(from 145deg at 48% 50%, rgba(255, 100, 100, 0.35), rgba(255, 200, 50, 0.38), rgba(80, 255, 150, 0.3), rgba(0, 200, 255, 0.42), rgba(180, 100, 255, 0.35), rgba(255, 100, 100, 0.35)); filter: blur(28px) contrast(140%) brightness(115%); opacity: 0.65; mix-blend-mode: screen; pointer-events: none; border-radius: 38% 62% 63% 37% / 41% 44% 56% 59%; z-index: 1; }}
    .prism-facets {{ position: absolute; width: 480px; height: 480px; left: 50%; top: 55%; transform: translate(-50%, -48%); background: linear-gradient(135deg, transparent 40%, rgba(255,255,255,0.18) 48%, rgba(255,255,255,0.3) 50%, transparent 52%), linear-gradient(45deg, transparent 42%, rgba(130,220,255,0.2) 49%, rgba(255,255,255,0.25) 50%, transparent 54%), linear-gradient(70deg, transparent 46%, rgba(255,180,240,0.18) 50%, transparent 53%); filter: blur(4px); opacity: 0.55; pointer-events: none; z-index: 1; }}
    .table-row-border {{ border-bottom: 1px solid rgba(255, 255, 255, 0.08); }}
    
    .chart-img {{ filter: drop-shadow(0 0 8px rgba(0, 242, 254, 0.4)) drop-shadow(0 0 15px rgba(74, 222, 128, 0.2)); border-radius: 12px; width: 100%; height: auto; }}
</style>
</head>
<body class="min-h-screen relative font-sans flex flex-col items-center p-4 md:p-8 space-y-16">

<!-- Fixed Background Canvas -->
<canvas class="fixed inset-0 w-full h-full pointer-events-none z-0 opacity-50" id="constellationCanvas"></canvas>

<!-- Top Background Ambient Lights -->
<div class="fixed -right-20 top-1/4 w-[480px] h-[480px] bg-emerald-500/10 rounded-full blur-[120px] pointer-events-none z-0"></div>
<div class="fixed left-1/3 top-1/3 w-[360px] h-[360px] bg-cyan-500/10 rounded-full blur-[100px] pointer-events-none z-0"></div>

<!-- HUD Metrics -->
<main class="relative z-10 w-full max-w-[1360px] bg-[#050b12] rounded-2xl overflow-hidden border border-cyan-950/40 shadow-2xl flex flex-col p-6 sm:p-10 mb-8 mt-4">
<h1 class="text-3xl md:text-5xl font-normal tracking-wide text-white drop-shadow-md text-center mb-8">AML Alert Prioritization</h1>
<section class="relative z-10 w-full flex-1 flex flex-wrap items-center justify-around px-2 py-8 my-auto">
<svg class="absolute inset-0 w-full h-full pointer-events-none z-0 hidden lg:block" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="cyberBeam" x1="0%" x2="100%" y1="0%" y2="0%">
<stop offset="0%" stop-color="#06b6d4" stop-opacity="0.2"></stop>
<stop offset="40%" stop-color="#10b981" stop-opacity="0.75"></stop>
<stop offset="70%" stop-color="#06b6d4" stop-opacity="0.85"></stop>
<stop offset="100%" stop-color="#4ade80" stop-opacity="0.9"></stop>
</linearGradient>
</defs>
<path d="M 220 150 C 350 150, 450 150, 500 150" fill="none" stroke="url(#cyberBeam)" stroke-dasharray="4 3" stroke-width="1.5"></path>
<path d="M 650 150 C 750 150, 850 150, 950 150" fill="none" stroke="url(#cyberBeam)" stroke-dasharray="3 2" stroke-width="1.5"></path>
</svg>

<div class="relative z-10 flex flex-col items-center flex-1 min-w-[200px] m-4">
<div class="relative w-40 h-40 sm:w-44 sm:h-44 flex items-center justify-center">
<svg class="absolute inset-0 w-full h-full animate-spin-reverse" viewBox="0 0 160 160">
<circle cx="80" cy="80" fill="none" r="72" stroke="#0f293d" stroke-width="2"></circle>
<path class="glow-cyan" d="M 80,8 A 72,72 0 1,1 25,126" fill="none" stroke="#06b6d4" stroke-linecap="round" stroke-width="3"></path>
<path d="M 80,18 A 62,62 0 0,1 142,80" fill="none" stroke="#34d399" stroke-linecap="round" stroke-width="3.5"></path>
<path d="M 35,120 A 62,62 0 0,0 80,142" fill="none" opacity="0.8" stroke="#22d3ee" stroke-linecap="round" stroke-width="2"></path>
</svg>
<div class="w-32 h-32 rounded-full border border-cyan-500/30 bg-[#07131d]/60 backdrop-blur-sm flex items-center justify-center shadow-[inset_0_0_15px_rgba(6,182,212,0.15)]">
<span class="font-hud text-3xl sm:text-4xl font-semibold text-white tracking-wider glow-text">{alerts}</span>
</div>
</div>
<p class="mt-4 text-[13px] sm:text-sm font-light text-slate-300/80 tracking-wide text-center">Тренировочных алертов</p>
</div>

<div class="relative z-10 flex flex-col items-center flex-1 min-w-[200px] m-4">
<div class="relative w-44 h-44 sm:w-48 sm:h-48 flex items-center justify-center">
<svg class="absolute inset-0 w-full h-full animate-spin-slow" viewBox="0 0 180 180">
<circle cx="90" cy="90" fill="none" opacity="0.6" r="82" stroke="#0ea5e9" stroke-dasharray="3 5 8 5" stroke-width="1.8"></circle>
<circle class="glow-cyan" cx="90" cy="90" fill="none" opacity="0.9" r="74" stroke="#10b981" stroke-dasharray="1 3" stroke-linecap="round" stroke-width="4.5"></circle>
<path d="M 90,16 A 74,74 0 0,1 164,90" fill="none" stroke="#34d399" stroke-linecap="round" stroke-width="4"></path>
<path d="M 40,140 A 74,74 0 0,0 90,164" fill="none" stroke="#06b6d4" stroke-width="2.5"></path>
</svg>
<svg class="absolute inset-0 w-full h-full animate-spin-reverse opacity-70" viewBox="0 0 180 180">
<circle cx="90" cy="90" fill="none" r="63" stroke="#22d3ee" stroke-dasharray="16 10 4 10" stroke-width="1.5"></circle>
<circle cx="90" cy="90" fill="none" r="54" stroke="#065f46" stroke-width="1"></circle>
</svg>
<div class="w-28 h-28 sm:w-32 sm:h-32 rounded-full border border-teal-400/40 bg-[#071720]/75 backdrop-blur-md flex items-center justify-center shadow-[inset_0_0_20px_rgba(20,184,166,0.2)]">
<span class="font-hud text-3xl sm:text-4xl font-semibold text-white tracking-wider glow-text">{transactions}</span>
</div>
</div>
<p class="mt-4 text-[13px] sm:text-sm font-light text-slate-300/80 tracking-wide text-center">Транзакций</p>
</div>

<div class="relative z-10 flex flex-col items-center flex-1 min-w-[200px] m-4">
<div class="relative w-40 h-40 sm:w-44 sm:h-44 flex items-center justify-center">
<svg class="absolute inset-0 w-full h-full animate-spin-slow" viewBox="0 0 160 160">
<circle cx="80" cy="80" fill="none" opacity="0.6" r="70" stroke="#064e3b" stroke-width="1.5"></circle>
<path class="glow-cyan" d="M 80,10 A 70,70 0 0,1 150,80" fill="none" stroke="#4ade80" stroke-linecap="round" stroke-width="3"></path>
<path d="M 80,150 A 70,70 0 0,1 10,80" fill="none" stroke="#06b6d4" stroke-linecap="round" stroke-width="2.5"></path>
</svg>
<svg class="absolute inset-0 w-full h-full animate-spin-reverse opacity-80" viewBox="0 0 160 160">
<circle cx="80" cy="80" fill="none" r="58" stroke="#10b981" stroke-dasharray="10 8" stroke-width="1.5"></circle>
</svg>
<div class="w-28 h-28 sm:w-30 sm:h-30 rounded-full border border-emerald-500/30 bg-[#06141a]/70 backdrop-blur-sm flex items-center justify-center shadow-[inset_0_0_15px_rgba(16,185,129,0.2)]">
<span class="font-hud text-3xl sm:text-4xl font-semibold text-white tracking-wider glow-text">{columns}</span>
</div>
</div>
<p class="mt-4 text-[13px] sm:text-sm font-light text-slate-300/80 tracking-wide text-center">Отобранных признаков</p>
</div>

<div class="relative z-10 flex flex-col items-center flex-[1.4] min-w-[280px] m-4">
<div class="relative w-64 h-64 sm:w-72 sm:h-72 flex items-center justify-center animate-pulse-glow">
<svg class="absolute inset-0 w-full h-full animate-spin-slow pointer-events-none" viewBox="0 0 280 280">
<circle class="glow-cyan-lg" cx="140" cy="140" fill="none" r="130" stroke="url(#vortexGrad)" stroke-dasharray="2 6 12 5 28 8" stroke-linecap="round" stroke-width="4.5"></circle>
<circle cx="140" cy="140" fill="none" opacity="0.6" r="122" stroke="#38bdf8" stroke-dasharray="1 4" stroke-width="1.5"></circle>
</svg>
<svg class="absolute inset-0 w-full h-full animate-spin-reverse pointer-events-none" viewBox="0 0 280 280">
<defs>
<linearGradient id="vortexGrad" x1="0%" x2="100%" y1="0%" y2="100%">
<stop offset="0%" stop-color="#4ade80"></stop>
<stop offset="45%" stop-color="#22d3ee"></stop>
<stop offset="85%" stop-color="#3b82f6"></stop>
<stop offset="100%" stop-color="#a855f7"></stop>
</linearGradient>
</defs>
<path class="glow-cyan" d="M 140,25 A 115,115 0 0,1 255,140" fill="none" stroke="#4ade80" stroke-linecap="round" stroke-width="4"></path>
<path d="M 255,140 A 115,115 0 0,1 140,255" fill="none" stroke="#22d3ee" stroke-dasharray="8 6" stroke-width="3"></path>
<circle cx="140" cy="140" fill="none" opacity="0.8" r="106" stroke="#2dd4bf" stroke-dasharray="30 14 10 14" stroke-width="2"></circle>
</svg>
<svg class="absolute inset-0 w-full h-full animate-spin-fast pointer-events-none" viewBox="0 0 280 280">
<circle cx="140" cy="140" fill="none" opacity="0.75" r="92" stroke="#67e8f9" stroke-dasharray="5 15" stroke-width="2"></circle>
</svg>
<div class="w-48 h-48 sm:w-52 sm:h-52 rounded-full border-2 border-emerald-400/60 bg-gradient-to-br from-[#06242c]/90 via-[#03151f]/90 to-[#020b12]/95 backdrop-blur-md flex flex-col items-center justify-center shadow-[inset_0_0_35px_rgba(52,211,153,0.35),0_0_25px_rgba(34,211,238,0.25)]">
<span class="font-hud text-5xl sm:text-6xl font-bold tracking-tight text-[#6ee7b7] glow-text">{auc}</span>
</div>
</div>
<p class="mt-2 text-sm sm:text-base font-medium text-slate-300 tracking-wider text-center">CV ROC-AUC</p>
</div>
</section>
<div class="relative z-10 w-full flex items-center justify-between text-[11px] text-slate-600 font-mono tracking-widest pt-2">
<span class="opacity-40">SYSTEM // TELEMETRY LINK ACTIVE</span>
<span class="opacity-40">QUANT_METRICS_V4.2</span>
</div>
</main>

<!-- EDA Section -->
<section class="relative z-10 w-full max-w-[1400px] flex flex-col items-center px-4">
<header class="w-full text-center mb-8 md:mb-12">
<h2 class="text-3xl md:text-4xl font-normal tracking-wide text-white drop-shadow-md">В поисках паттернов: Разведочный анализ (EDA)</h2>
</header>
<div class="w-full grid grid-cols-1 lg:grid-cols-2 gap-8 md:gap-10">
<article class="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch">
<div class="md:col-span-7 flex flex-col justify-end">
  <div class="relative w-full"><img src="{weekly}" class="chart-img" alt="Weekly transaction volume"></div>
</div>
<div class="md:col-span-5 glass-card rounded-2xl p-6 sm:p-7 flex flex-col justify-center">
  <h3 class="text-white text-lg font-medium mb-4">Активность во времени</h3>
  <p class="text-slate-400 text-sm leading-relaxed mb-4">Еженедельный объем транзакций. Ясно видно распределение активности.</p>
</div>
</article>

<article class="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch">
<div class="md:col-span-7 flex flex-col justify-end">
  <div class="relative w-full"><img src="{direction_types}" class="chart-img" alt="Direction and type distributions"></div>
</div>
<div class="md:col-span-5 glass-card rounded-2xl p-6 sm:p-7 flex flex-col justify-center">
  <h3 class="text-white text-lg font-medium mb-4">Направления и типы транзакций</h3>
  <p class="text-slate-400 text-sm leading-relaxed mb-4">Показывает распределение входящих и исходящих переводов, а также классификацию по типам.</p>
</div>
</article>

<article class="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch">
<div class="md:col-span-5 order-2 md:order-1 glass-card rounded-2xl p-6 sm:p-7 flex flex-col justify-center">
  <h3 class="text-white text-lg font-medium mb-4 leading-snug">Всплеск активности перед алертом</h3>
  <p class="text-slate-400 text-sm leading-relaxed">Плотность распределения активности в дни, предшествующие алерту. Эскалированные кейсы показывают более высокую концентрацию в последние дни.</p>
</div>
<div class="md:col-span-7 order-1 md:order-2 flex flex-col justify-end">
  <div class="relative w-full"><img src="{days_before}" class="chart-img" alt="Activity before alert"></div>
</div>
</article>

<article class="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch">
<div class="md:col-span-7 flex flex-col justify-end">
  <div class="relative w-full"><img src="{type_outcome}" class="chart-img" alt="Mix by outcome"></div>
</div>
<div class="md:col-span-5 glass-card rounded-2xl p-6 sm:p-7 flex flex-col justify-center">
  <h3 class="text-white text-lg font-medium mb-4 leading-snug">Типы транзакций в зависимости от исхода</h3>
  <p class="text-slate-400 text-sm leading-relaxed mb-4">Доля каждого типа транзакций в разрезе финального статуса алерта (Dismissed vs Escalated).</p>
</div>
</article>

<!-- Additional Target Distribution -->
<article class="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch lg:col-span-2">
<div class="md:col-span-6 flex flex-col justify-end">
  <div class="relative w-full"><img src="{target}" class="chart-img" alt="Target distribution"></div>
</div>
<div class="md:col-span-6 glass-card rounded-2xl p-6 sm:p-7 flex flex-col justify-center">
  <h3 class="text-white text-lg font-medium mb-4 leading-snug">Распределение таргета</h3>
  <p class="text-slate-400 text-sm leading-relaxed mb-4">Большинство алертов закрываются (Dismissed). Доля эскалированных (Escalated) составляет {rate}.</p>
</div>
</article>
</div>
</section>

<!-- Table Section -->
<section class="relative z-10 w-full max-w-[1024px] min-h-[559px] bg-[#070b10] rounded-lg shadow-2xl overflow-hidden flex flex-col justify-between p-7 border border-slate-800/40 mt-8 mb-16">
<div aria-hidden="true" class="prism-background"></div>
<div aria-hidden="true" class="prism-facets"></div>

<header class="relative z-10 flex flex-col items-center w-full">
<h2 class="text-[32px] sm:text-[34px] font-normal tracking-wide text-white text-center mb-5">Больше данных ≠ Лучше. Отсечение шума.</h2>
<aside class="w-full max-w-[850px] bg-[#0e161c]/80 backdrop-blur-md border border-[#4ade80]/90 rounded-xl px-5 py-3 shadow-[0_0_15px_rgba(74,222,128,0.15)] flex items-start gap-3.5">
<span aria-hidden="true" class="text-[#4ade80] text-2xl font-serif font-black leading-none mt-0.5 select-none">“</span>
<p class="text-[13.5px] leading-[1.4] text-slate-200 font-normal">
  Мы измерили семейства признаков строгой кросс-валидацией. Точность достигла пика на компактном наборе. Добавление остальных только снижало ROC-AUC. Из отвергнутых: {rejected_families}, из них на уровне случайности: {chance_families}. Drift: {drift_verdict} (Adversarial AUC {adversarial}).
</p>
</aside>
</header>

<div class="relative z-10 w-full flex justify-center pb-2 mt-8">
<div class="w-full max-w-[725px] bg-[#0c1219]/75 backdrop-blur-xl border border-white/10 rounded-2xl p-4 shadow-[0_20px_50px_rgba(0,0,0,0.6)]">
<table class="w-full text-left text-sm border-collapse">
<thead>
<tr class="table-row-border text-slate-300 font-medium text-[13px]">
<th class="pb-2.5 pl-4 font-normal text-slate-300 w-[44%]" scope="col">Family added</th>
<th class="pb-2.5 text-center font-normal text-slate-300 w-[16%]" scope="col">Columns</th>
<th class="pb-2.5 text-center font-normal text-slate-300 w-[20%]" scope="col">
<span class="inline-flex items-center gap-2"><span class="text-white/20 text-xs">|</span> CV mean <span class="text-white/20 text-xs">|</span></span>
</th>
<th class="pb-2.5 pr-4 text-center font-normal text-slate-300 w-[20%]" scope="col">Decision</th>
</tr>
</thead>
<tbody class="divide-y divide-white/[0.06] text-[13px] font-normal tracking-tight text-slate-300">
{ablation_rows}
</tbody>
</table>
</div>
</div>
</section>

<script>
    (function initConstellation() {{
      const canvas = document.getElementById('constellationCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      let width, height;
      let nodes = [];

      function resize() {{
        width = canvas.width = window.innerWidth;
        height = canvas.height = document.body.scrollHeight;
        initNodes();
      }}

      function initNodes() {{
        nodes = [];
        const nodeCount = Math.floor((width * window.innerHeight) / 22000) + 35;
        for (let i = 0; i < nodeCount; i++) {{
          nodes.push({{
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.35,
            vy: (Math.random() - 0.5) * 0.35,
            radius: Math.random() * 1.5 + 1
          }});
        }}
      }}

      function draw() {{
        ctx.clearRect(0, 0, width, height);

        ctx.strokeStyle = 'rgba(70, 95, 140, 0.15)';
        ctx.lineWidth = 0.8;
        for (let i = 0; i < nodes.length; i++) {{
          for (let j = i + 1; j < nodes.length; j++) {{
            const dx = nodes[i].x - nodes[j].x;
            const dy = nodes[i].y - nodes[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < 140) {{
              const alpha = (1 - dist / 140) * 0.25;
              ctx.strokeStyle = `rgba(56, 189, 248, ${{alpha}})`;
              ctx.beginPath();
              ctx.moveTo(nodes[i].x, nodes[i].y);
              ctx.lineTo(nodes[j].x, nodes[j].y);
              ctx.stroke();
            }}
          }}
        }}

        for (let i = 0; i < nodes.length; i++) {{
          const n = nodes[i];
          n.x += n.vx;
          n.y += n.vy;

          if (n.x < 0 || n.x > width) n.vx *= -1;
          if (n.y < 0 || n.y > height) n.vy *= -1;

          ctx.fillStyle = 'rgba(148, 163, 184, 0.4)';
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
          ctx.fill();
        }}

        requestAnimationFrame(draw);
      }}

      window.addEventListener('resize', resize);
      setTimeout(resize, 500);
      draw();
    }})();
</script>
</body>
</html>"""
