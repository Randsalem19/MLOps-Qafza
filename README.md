# MLOps Engineering Training Portfolio

### Qafza Tech — Applied MLOps Scholarship (2026/2027)

![Status](https://img.shields.io/badge/status-in--progress-yellow)
![Program](https://img.shields.io/badge/program-Qafza%20MLOps%20Scholarship-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PostgreSQL](https://img.shields.io/badge/database-PostgreSQL%2017-336791)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

This repository documents my assignments and practical work throughout the **Qafza Tech MLOps Engineering Scholarship**, a 12‑week, hands‑on program covering the full MLOps lifecycle — from data engineering to production deployment.

The project builds a reproducible, end‑to‑end machine learning workflow on the **Brazilian E‑Commerce Public Dataset by Olist**, predicting whether an order will be delivered **on time or late**.

---

## Author

**Rand Majed Salem**
AI and Machine Learning Student — University College of Applied Sciences (UCAS)
Scholarship Program: Qafza Tech, MLOps Engineering Track

---

## Program Roadmap

The scholarship follows a 12‑week roadmap plus a final capstone. Progress is tracked below as each week's task is completed.

| Week | Phase | Topic | Core Tools | Status |
|---|---|---|---|---|
| 1 | Local Foundations | Leakage‑proof ML Pipeline | Python, Scikit-learn | ✅ Completed — [Task 01](Tasks/Task-01) · [Task 02](Tasks/Task-02) |
| 2 | Local Foundations | Deep Learning Pipeline | PyTorch, Hugging Face | ⬜ Upcoming |
| 3 | Production APIs | Production API | FastAPI, Pydantic | ⬜ Upcoming |
| 4 | Containerization | Docker | Docker Engine | ⬜ Upcoming |
| 5 | Data Pipelines | ETL Pipeline | Python ETL, Database | ⬜ Upcoming |
| 6 | Data Versioning | Versioning | DVC | ⬜ Upcoming |
| 7 | Experiment Tracking | MLflow | MLflow Registry | ⬜ Upcoming |
| 8 | Distributed Training | Distributed ML | Ray Core, Ray Train | ⬜ Upcoming |
| 9 | Feature Store | Feature Management | Feast, Ray | ⬜ Upcoming |
| 10 | Monitoring | Monitoring | Prometheus, Grafana | ⬜ Upcoming |
| 11 | Continuous Retraining | Automation | Ray Serve, GitHub Actions | ⬜ Upcoming |
| 12 | Infrastructure as Code | Infrastructure | Terraform | ⬜ Upcoming |
| Capstone | Capstone | End‑to‑End ML System | Complete Stack | ⬜ Upcoming |

---

## Repository Structure

```text
MLOps-Qafza/
│
├── Tasks/
│   ├── Task-01/                     # Database ingestion & verification
│   │   ├── data/
│   │   ├── notebooks/
│   │   ├── screenshots/
│   │   ├── scripts/
│   │   ├── sql/
│   │   └── README.md
│   │
│   └── Task-02/                     # End-to-end ML notebook pipeline
│       ├── artifacts/
│       ├── figures/
│       ├── notebooks/
│       ├── reports/
│       ├── .env.example
│       ├── requirements.txt
│       └── README.md
│
├── .gitignore
└── README.md
```

---

## Completed Tasks

### Task 01 — Olist Database Ingestion

Loaded the full Olist relational dataset (9 CSV files) into a local PostgreSQL database, defined keys/constraints, and verified the load with SQL queries and an automated Python check.

- Nine relational tables loaded and validated (customers, orders, order_items, order_payments, order_reviews, products, sellers, geolocation, product category translation)
- Primary/foreign keys and indexes defined in [`sql/01_schema.sql`](Tasks/Task-01/sql/01_schema.sql)
- Automated row-count and join verification via [`scripts/verify_data.py`](Tasks/Task-01/scripts/verify_data.py)
- Target sanity check: **7,826 late orders** vs **88,644 on‑time orders** out of 96,470 delivered orders

📄 Full details: [Tasks/Task-01/README.md](Tasks/Task-01/README.md)

### Task 02 — From PostgreSQL Tables to ML Notebooks

Built a leakage‑safe, reproducible ML pipeline across six ordered notebooks: table joining, label creation, chronological train/validation/test split, training‑only EDA, feature engineering, and model training/tuning/evaluation.

| Split | Rows | Late rate | Date range |
|---|---:|---:|---|
| Train | 67,529 | 9.03% | Sep 2016 – Apr 2018 |
| Validation | 14,470 | 5.34% | Apr 2018 – Jun 2018 |
| Test | 14,471 | 6.61% | Jun 2018 – Aug 2018 |

**Test set results** (evaluated once, after all tuning decisions):

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | PR‑AUC | ROC‑AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy (most‑frequent) | 93.4% | 50.0% | 0.00 | 0.00 | 0.00 | 0.066 | 0.500 |
| Class‑weighted Logistic Regression | 77.3% | 56.1% | 0.103 | 0.318 | 0.156 | 0.121 | 0.683 |

Key design decisions: one row per order (item/payment tables pre‑aggregated), post‑delivery fields and reviews excluded to prevent leakage, chronological 70/15/15 split to mimic production, and PR‑AUC used as the primary metric due to class imbalance (late orders ≈ 9% of the data).

📄 Full details: [Tasks/Task-02/README.md](Tasks/Task-02/README.md)

---

## Tech Stack

`Python` `PostgreSQL` `pgAdmin` `Pandas` `Scikit-learn` `Jupyter` `Parquet`

---

## How to Explore This Repository

1. Start with [`Tasks/Task-01`](Tasks/Task-01) to see how the raw dataset was ingested and verified.
2. Continue to [`Tasks/Task-02`](Tasks/Task-02) for the leakage‑safe ML pipeline, from raw tables to a trained, evaluated model.
3. Each task folder has its own `README.md` with setup instructions, design rationale, and results.

> Datasets, credentials (`.env`), and large generated artifacts are excluded from version control — see each task's `.gitignore` and `.env.example`.

---

## License

This repository is shared for educational and portfolio purposes as part of the Qafza Tech MLOps Scholarship.
