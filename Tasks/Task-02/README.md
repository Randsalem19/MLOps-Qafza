# Task 02 - From PostgreSQL Tables to ML Notebooks

**Author:** Rand Majed Salem  
**Program:** Qafza Tech MLOps Training 2026/2027

## Objective

Build a reproducible, leakage-safe machine-learning workflow for predicting late delivery from the Olist PostgreSQL database. The task is intentionally divided into six ordered notebooks, with each notebook reading the artifacts produced by the previous step and writing artifacts for the next step.

## Workflow

| # | Notebook | Responsibility | Main artifact |
|---:|---|---|---|
| 1 | `01_read_and_join_tables.ipynb` | Read all tables, inspect grain and keys, aggregate one-to-many tables, join correctly | `01_ml_table.parquet` |
| 2 | `02_create_labels.ipynb` | Create and validate `is_late`; quantify imbalance | `02_labeled_orders.parquet` |
| 3 | `03_train_validation_test_split.ipynb` | Chronological 70/15/15 split | Three split Parquet files |
| 4 | `04_eda_training_only.ipynb` | Detailed EDA using training data only | Figures and findings summary |
| 5 | `05_feature_engineering.ipynb` | Fit preprocessing on train only; transform all splits | Sparse matrices, transformer, feature list |
| 6 | `06_train_tune_evaluate.ipynb` | Baseline, validation tuning, one-time test evaluation | Model and results summary |

## Important Design Decisions

### One row per order

`order_items` and `order_payments` contain multiple rows per order. They are aggregated before joining, preventing accidental row multiplication.

### Reviews are excluded from model inputs

Reviews are created after delivery. They are inspected in Notebook 1 but never joined into the prediction table because they would leak future information.

### Time-based split

Orders are sorted by `order_purchase_timestamp`, then split chronologically into 70% train, 15% validation, and 15% test. This matches the production setting: learn from past orders and predict future orders.

### Leakage-safe features

The model excludes:

- Actual customer delivery date
- Carrier delivery date
- Delivery delay
- Order status
- Reviews
- Any other post-delivery information

The estimated delivery date is used only through the promised delivery-window length, which is available at order time.

### Imbalanced evaluation

Late orders are the minority class. Model selection uses PR-AUC rather than accuracy alone. F1, recall, precision, balanced accuracy, and ROC-AUC are also reported.

## Setup on Windows

Open CMD inside `Tasks\Task-02`.

### 1. Create the environment

```cmd
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Configure PostgreSQL

```cmd
copy .env.example .env
notepad .env
```

Enter the same PostgreSQL credentials used in Task 1. The example uses port `2003`, matching the local setup used during Task 1. Never upload `.env` to GitHub.

### 3. Launch Jupyter

```cmd
.venv\Scripts\python.exe -m jupyter notebook
```

Open the `notebooks` folder and run every notebook using **Kernel > Restart Kernel and Run All Cells**, strictly in order from 01 through 06.

## Expected Artifact Flow

```text
PostgreSQL tables
  -> 01_ml_table.parquet
  -> 02_labeled_orders.parquet
  -> 03_train.parquet + 03_validation.parquet + 03_test.parquet
  -> EDA figures and findings
  -> fitted preprocessor + feature matrices + feature names
  -> trained model + test results
```

## Model Strategy

- Baseline: most-frequent `DummyClassifier`
- Candidate: class-weighted logistic regression
- Hyperparameter selection: validation PR-AUC
- Threshold selection: validation F1
- Final model fit: train + validation
- Test evaluation: exactly once, after all decisions

This model is deliberately interpretable and reproducible. The task asks for a first result and a baseline comparison, not the most complex possible model.

## Completion Checklist

- [ ] Notebook 1 runs from PostgreSQL and saves one row per order.
- [ ] Notebook 2 validates real label examples and saves the labeled table.
- [ ] Notebook 3 saves non-overlapping chronological splits.
- [ ] Notebook 4 reads training data only and saves charts/findings.
- [ ] Notebook 5 fits transformations on train only and saves every fitted object.
- [ ] Notebook 6 compares against a baseline and evaluates test once.
- [ ] All notebooks run in order from a clean environment.
- [ ] Screenshots/results are added after local execution.

## GitHub Note

Large generated artifacts are ignored by Git. Code, notebooks, reports, and figures remain trackable. Do not upload database passwords or the raw Olist dataset.

