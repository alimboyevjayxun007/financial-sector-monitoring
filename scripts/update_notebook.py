import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src import config
from src.data_loading import load_signals, load_transactions
from src.features import build
from src.model import train, cross_validate
from src.predict import predict, validate_submission, write

TEAM_ID = "C6FD20A0"

print("Loading raw data...")
train_signals = load_signals(config.TRAIN_SIGNALS_PATH)
train_transactions = load_transactions(config.TRAIN_TRANSACTIONS_PATH)
test_signals = load_signals(config.TEST_SIGNALS_PATH)
test_transactions = load_transactions(config.TEST_TRANSACTIONS_PATH)

shapes_str = str((train_signals.shape, train_transactions.shape, test_signals.shape, test_transactions.shape))
print("Shapes:", shapes_str)

print("Building features...")
train_features = build(train_signals, train_transactions)
test_features = build(test_signals, test_transactions)

X = train_features[config.FEATURE_COLUMNS]
y = train_features[config.TARGET_COL]

print("Cross-validating...")
auc = cross_validate(X, y)
auc_str = f"CV ROC-AUC: {auc:.4f}"
print(auc_str)

print("Training final model...")
model = train(X, y)

print("Predicting...")
predictions = predict(model, test_features)
validate_submission(predictions, test_signals[config.ID_COL])
out_path = str(config.OUTPUTS_DIR / f"team_{TEAM_ID}.csv")
write(predictions, out_path)
print(f"wrote {out_path}")

head_df = predictions.head()
head_html = head_df.to_html()
head_text = head_df.to_string()

nb_path = config.ROOT / "notebooks" / "submission_pipeline.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "id": "035aee2b",
        "metadata": {},
        "source": [
            "# AML Signal Escalation — Submission Pipeline\n",
            "\n",
            "Track A (feature engineering) va Track B (modeling) qismlarini boshidan oxirigacha birlashtiruvchi to'liq qayta ishga tushiriluvchi (reproducible) notebook.\n",
            "Jamoa ID: **C6FD20A0** | Baholash metrikasi: **ROC-AUC**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "id": "e7271ac0",
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys, pathlib\n",
            "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n",
            "\n",
            "TEAM_ID = \"C6FD20A0\"\n",
            "\n",
            "from src import config\n",
            "from src.data_loading import load_signals, load_transactions\n",
            "from src.features import build\n",
            "from src.model import train, cross_validate\n",
            "from src.predict import predict, validate_submission, write"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "c03c89c7",
        "metadata": {},
        "source": [
            "## 1. Load raw data"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "id": "5e69829e",
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "text/plain": [
                        shapes_str
                    ]
                },
                "execution_count": 2,
                "metadata": {},
                "output_type": "execute_result"
            }
        ],
        "source": [
            "train_signals = load_signals(config.TRAIN_SIGNALS_PATH)\n",
            "train_transactions = load_transactions(config.TRAIN_TRANSACTIONS_PATH)\n",
            "test_signals = load_signals(config.TEST_SIGNALS_PATH)\n",
            "test_transactions = load_transactions(config.TEST_TRANSACTIONS_PATH)\n",
            "train_signals.shape, train_transactions.shape, test_signals.shape, test_transactions.shape"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "286f911a",
        "metadata": {},
        "source": [
            "## 2. Build features (Track A: 25 domain features, 0% lookahead leakage)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "id": "3b6346e6",
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "text/html": [
                        train_features.head(3).to_html()
                    ],
                    "text/plain": [
                        train_features.head(3).to_string()
                    ]
                },
                "execution_count": 3,
                "metadata": {},
                "output_type": "execute_result"
            }
        ],
        "source": [
            "train_features = build(train_signals, train_transactions)\n",
            "test_features = build(test_signals, test_transactions)\n",
            "train_features.head(3)"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "7dcfa1ff",
        "metadata": {},
        "source": [
            "## 3. Cross-validate and train model (Track B: Calibrated Logistic Regression)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "id": "426be2a8",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    f"{auc_str}\n"
                ]
            }
        ],
        "source": [
            "X = train_features[config.FEATURE_COLUMNS]\n",
            "y = train_features[config.TARGET_COL]\n",
            "auc = cross_validate(X, y)\n",
            "print(f\"CV ROC-AUC: {auc:.4f}\")\n",
            "model = train(X, y)"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "0e860245",
        "metadata": {},
        "source": [
            "## 4. Predict + write submission (outputs/team_C6FD20A0.csv)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "id": "c414cee1",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    f"wrote {out_path}\n"
                ]
            },
            {
                "data": {
                    "text/html": [
                        head_html
                    ],
                    "text/plain": [
                        head_text
                    ]
                },
                "execution_count": 5,
                "metadata": {},
                "output_type": "execute_result"
            }
        ],
        "source": [
            "predictions = predict(model, test_features)\n",
            "validate_submission(predictions, test_signals[config.ID_COL])\n",
            "out_path = str(config.OUTPUTS_DIR / f\"team_{TEAM_ID}.csv\")\n",
            "write(predictions, out_path)\n",
            "print(f\"wrote {out_path}\")\n",
            "predictions.head()"
        ]
    }
]

notebook_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14.4"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_data, f, indent=1, ensure_ascii=False)

print(f"Successfully generated pristine reproducible notebook at {nb_path}!")
