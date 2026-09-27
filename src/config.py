from pathlib import Path

SEED = 42

# LightGBM and XGBoost are deterministic for a fixed thread count, not across
# thread counts. n_jobs=-1 resolves to the host core count, so the same seed on
# a different machine would produce different trees and a different selection.
N_JOBS = 4

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fintech_data"
EXPERIMENTS = ROOT / "experiments"
OUTPUTS = ROOT / "outputs"
SITE_DATA = ROOT / "site_data"
DOCS = ROOT / "docs"

TRAIN_SIGNALS = DATA / "train_signals.csv"
TEST_SIGNALS = DATA / "test_signals.csv"
TRAIN_TX = DATA / "train_transactions.parquet"
TEST_TX = DATA / "test_transactions.parquet"
SAMPLE_SUBMISSION = DATA / "sample_submission (3).csv"

SUBMISSION_NAME = "team_2ABB3C78.csv"

DIRECTIONS = ("kirim", "chiqim")
TX_TYPES = ("karta", "bank_otkazmasi", "naqd", "xalqaro")

TARGET = "eskalatsiya"
ID = "signal_id"
PROBA = "ehtimollik"
