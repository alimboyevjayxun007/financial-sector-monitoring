# AML Alert Prioritization

> **WIUT Hackathon 2026 · FinTech / AI in Finance**  
> **Team:** DnkCode (`2ABB3C78`)  
> **Submission deadline:** Sunday, 27 September 2026, 23:59 (Tashkent time)

## Challenge

A financial monitoring team reviews automated alerts based on customers' transaction histories. Each alert is either dismissed or escalated for further investigation.

Build a model that estimates the probability that an alert will be escalated. The competition dataset is synthetic, transformed, and localized for this hackathon; it contains no real customer or transaction records from the Central Bank of Uzbekistan or any commercial bank.

## Prediction Target

For every `signal_id` in `test_signals.csv`, predict the probability of escalation. Submit a numeric score between `0` and `1` for each alert.

**Evaluation metric:** ROC-AUC. Submit probability scores rather than hard class labels.

## Dataset

The supplied files are in `fintech_data/`.

| File | Contents |
| --- | --- |
| `train_signals.csv` | Training alerts and their escalation labels |
| `train_transactions.parquet` | Transaction history associated with training alerts |
| `test_signals.csv` | Test alerts without labels |
| `test_transactions.parquet` | Transaction history associated with test alerts |
| `sample_submission (3).csv` | Example prediction file showing the required columns and test IDs |

### Signal columns

| Column | Meaning |
| --- | --- |
| `signal_id` | Unique alert identifier |
| `signal_sanasi` | Date of the alert |
| `eskalatsiya` | Training target: `1` = escalated, `0` = dismissed |

`eskalatsiya` appears only in `train_signals.csv`.

### Transaction columns

| Column | Meaning |
| --- | --- |
| `signal_id` | Alert associated with the transaction |
| `tranzaksiya_vaqti` | Transaction timestamp |
| `kirim_chiqim` | Direction: `kirim` (incoming) or `chiqim` (outgoing) |
| `tranzaksiya_turi` | Type: `karta`, `bank_otkazmasi`, `naqd`, or `xalqaro` |
| `miqdor_indeksi` | Standardized transaction-size indicator |

## Modeling Considerations

Each alert can have many associated transactions. Summarize transaction behavior at the alert level or use another suitable approach. Potential features include:

- transaction counts, amounts, and history duration;
- incoming and outgoing activity, including their proportions;
- transaction-type counts and amount summaries;
- activity close to the alert date;
- differences in behavior between escalated and dismissed training alerts.

Use only information that would be available for the alert being scored. Do not use hidden labels, external copies of target labels, or organizer-only data.

## Required EDA Website

Prepare a small public website presenting the team's exploratory data analysis and the findings that informed its approach. The site must be reachable through a working public URL without requiring organizers to access a private account.

Include:

- a brief description of the approach and dataset;
- several useful visualizations of the transaction history;
- target-distribution and alert-behavior analysis;
- key findings and the modeling ideas they informed;
- a short conclusion.

Possible visualizations include transaction activity over time, incoming versus outgoing behavior, transaction types and sizes, activity before an alert, and behavior by escalation outcome. The website is a separate deliverable; it does not replace the prediction CSV.

## Submission

Submit all three deliverables:

1. Prediction file: `team_2ABB3C78.csv`
2. A working public URL for the EDA website
3. A reproducible Jupyter notebook

The prediction CSV must contain exactly two columns, in this order:

```csv
signal_id,ehtimollik
SG_000001,0.1842
SG_000002,0.8271
SG_000003,0.0537
```

Submission requirements:

- include exactly one row for every test `signal_id`;
- preserve the test IDs and do not include duplicates, missing IDs, or extra IDs;
- provide a non-missing `ehtimollik` value between `0` and `1` for every row;
- do not include a CSV index column;
- use the filename `team_2ABB3C78.csv`.

Late submissions will not be assessed.

## Local File Snapshot

| File | Rows | Columns |
| --- | ---: | ---: |
| `train_signals.csv` | 14,000 | 3 |
| `test_signals.csv` | 6,000 | 2 |
| `sample_submission (3).csv` | 6,000 | 2 |
| `train_transactions.parquet` | 6,987,663 | 5 |
| `test_transactions.parquet` | Present locally; row count not confirmed | 5 expected |

**Local data note:** `test_transactions.parquet` did not load successfully during the initial inspection. Verify it before using it for modeling.

## Questions

Send task questions to the organizers. The team's reference is attached automatically on the official platform.
