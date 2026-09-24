# Task 3 — From Notebooks to Production

Turns the leakage-safe pipeline from [Task 2](../Task-02) into a real inference
service: a config-driven repo, an inference pipeline that loads the fitted
objects (never re-fits), input validation, logging, tests, a FastAPI service,
Docker Compose, CI/CD, and monitoring.

Training stays in the Task 2 notebooks — this task is the **inference** side
only: input a new order, output `late` or `on_time` with a probability.

## Repository structure

```text
Task-03/
├── app/
│   └── main.py                  # FastAPI app: health, model info, predict, batch, metrics
├── src/inference_service/
│   ├── config.py                 # loads config/config.yaml once
│   ├── logging_config.py         # structured logging to console + rotating file
│   ├── schemas.py                 # pydantic request/response models
│   ├── features.py                # mirrors Notebook 05's build_feature_frame exactly
│   ├── validation.py              # fast pandas checks + a real Great Expectations suite
│   ├── model_loader.py            # loads from MLflow registry, falls back to local joblib
│   ├── predictor.py                # orchestrates validate -> features -> transform -> predict
│   └── monitoring.py               # Prometheus metrics + JSON-lines prediction log
├── scripts/
│   └── register_model.py          # combines Task-02's 2 joblib files into 1 MLflow model
├── config/
│   ├── config.yaml                # single source of truth for paths/features/thresholds
│   └── expectations.json          # validation rules, built from Task-02's own EDA stats
├── tests/                          # unit + data + model + integration tests (pytest)
├── models/                         # DVC-tracked model artifacts (not committed to git directly)
├── Dockerfile
├── docker-compose.yml              # postgres + mlflow (registry/artifact store) + api
├── .github/workflows/ci.yml        # lint, format check, tests, image build on every push
├── .pre-commit-config.yaml
├── requirements.txt / requirements-dev.txt
└── .env.example
```

## Getting the model artifacts

Task 2's notebooks save the fitted preprocessor and model as two separate
`joblib` files, gitignored because of size. Before running this service:

```bash
cp ../Task-02/artifacts/05_preprocessor.joblib models/
cp ../Task-02/artifacts/06_logistic_regression.joblib models/
cp ../Task-02/artifacts/05_feature_names.json models/
cp ../Task-02/reports/06_results_summary.json models/
```

These four files are what `models/*.dvc` point to (see **Data & model
versioning** below) — commit the `.dvc` files, never the artifacts themselves.

## Running locally (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # edit if needed; defaults work for local dev
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for interactive API docs, or:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "order_id": "example-0001",
  "order_purchase_timestamp": "2018-05-14T10:32:00",
  "order_estimated_delivery_date": "2018-05-29T00:00:00",
  "customer_zip_code_prefix": 14409, "item_count": 1, "product_count": 1, "seller_count": 1,
  "price_total": 89.9, "price_mean": 89.9, "freight_total": 16.11, "freight_mean": 16.11,
  "product_weight_mean": 700.0, "product_length_mean": 25.0, "product_height_mean": 13.0,
  "product_width_mean": 20.0, "product_photos_mean": 2.0, "seller_lat_mean": -23.55,
  "seller_lng_mean": -46.63, "seller_state_nunique": 1, "payment_count": 1,
  "payment_value_total": 106.01, "payment_installments_max": 2, "payment_type_count": 1,
  "customer_lat": -21.17, "customer_lng": -47.81, "customer_seller_distance_km": 248.5,
  "customer_state": "SP", "primary_seller_state": "SP",
  "primary_category": "bed_bath_table", "dominant_payment_type": "credit_card"
}
JSON
```

## Running with Docker Compose

```bash
docker compose up --build
```

Brings up three services together on a clean machine: `postgres` (the Task-01
database), `mlflow` (a real tracking server + model registry + artifact
store, reachable at `http://localhost:5000`), and `api` (this service, at
`http://localhost:8000`). The API falls back to the local `models/` files
until a model is registered — see below.

## Registering the model with MLflow (recommended)

Running from `models/*` directly works, but the intended path loads from the
registry — not from a file that only exists on one laptop:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000   # or sqlite:///mlflow.db for local-only
python scripts/register_model.py
```

This combines the two Task-02 joblib files into one `sklearn.Pipeline`, logs
it as an MLflow run (with the tuned threshold, `best_C`, and test metrics
attached), registers it under `order-delivery-classifier`, and points the
`production` alias at it. `model_loader.py` tries this registry first on
every startup and only falls back to `models/` if it's unreachable or empty.

## API routes

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | liveness/readiness + whether the model loaded |
| `/model/info` | GET | model name, version, decision threshold, feature count |
| `/predict` | POST | score one order (fast pandas validation) |
| `/predict/batch` | POST | score up to 1000 orders (full Great Expectations suite) |
| `/metrics` | GET | Prometheus scrape endpoint |

## Data validation — two tracks, one source of truth

Both read `config/expectations.json`, which was **generated from Task-02's
own EDA output** (`reports/04_numerical_summary.csv` / `04_categorical_summary.csv`),
not guessed:

- **Fast path** (`validate_fast`, used on `/predict`): plain pandas
  not-null / range / allowed-category checks. Sub-millisecond.
- **Great Expectations path** (`validate_with_great_expectations`, used on
  `/predict/batch`): a real GX 1.x `ExpectationSuite`, built once and reused.

Why two tracks instead of just GX everywhere: GX 1.x's fluent API rebuilds a
metrics graph on every `.validate()` call — roughly 150–250ms even for a
single row, even with the Data Source/Asset/Suite built once and reused. That
is fine for a nightly data-quality job or a batch of hundreds of orders, but
too slow for a single-order endpoint a caller expects to respond in a few
milliseconds. Both paths are generated from the same file, so they can never
silently disagree about what "valid" means.

On failure, the service returns `422` with the specific list of failed checks
(the `on_failure: reject` policy in `config.yaml` — see that file to switch
to `flag` behavior instead).

## Logging & monitoring

- Every request is logged (method, path, status, latency) via a FastAPI
  middleware, to console and to a rotating JSON-lines file (`logs/service.log`).
- Every prediction is separately logged to `logs/predictions.jsonl` with the
  full input, output, and latency — join this against the real
  `order_delivered_customer_date` once it's known to measure drift/decay over
  time.
- `/metrics` exposes Prometheus counters/histograms: request count by
  route+status, prediction count by outcome, prediction latency, and
  validation failure count. Point a local Prometheus + Grafana at this route
  to graph them.
- **What we'd alert on:** validation failure rate above ~5% of traffic
  (upstream data problem), p95 prediction latency above 500ms (compute or
  registry issue), and the ratio of `late` predictions drifting more than a
  few points away from the ~9% base rate seen in training (population drift).

## Testing

```bash
pytest tests/ -v --cov=src/inference_service
```

- `test_features.py` — the datetime-derived columns match Notebook 05 exactly
- `test_validation.py` — data tests: schema, ranges, nulls, both validation paths
- `test_predictor.py` — model tests: loads, predicts the right shape,
  deterministic on the same input, never re-fits the preprocessor
- `test_api.py` — integration tests for every route end to end

Tests never touch the real (large, gitignored) Task-02 artifacts — `conftest.py`
fits a tiny synthetic model with the exact same schema so the suite is fast
and runs the same on any machine, including CI.

## Data & model versioning (DVC)

```bash
dvc init
dvc remote add -d storage <your-remote-url>   # e.g. an S3 bucket, or a local path for now
dvc add models/05_preprocessor.joblib models/06_logistic_regression.joblib \
        models/05_feature_names.json models/06_results_summary.json
git add models/*.dvc models/.gitignore .dvc/config
git commit -m "Track model artifacts with DVC"
dvc push
```

Commit the small `.dvc` pointer files (already present in `models/`), never
the artifacts themselves. Anyone who clones the repo runs `dvc pull` to get
the exact model version a given commit was built against.

## CI/CD

`.github/workflows/ci.yml` runs on every push: `ruff check`, `ruff format
--check`, then the full pytest suite with coverage. The image-build job only
runs after that job succeeds, and only on `main`/`master` — a failing test
stops the pipeline before anything gets built.

`.pre-commit-config.yaml` runs the same lint/format/test checks locally
before a commit even reaches GitHub:

```bash
pre-commit install
```

## Reproducing the notebook's output

The whole point of this task: feed the API the same row Notebook 05 would
have engineered, and get the same prediction. `features.py`'s
`build_feature_frame` is copied 1:1 from Notebook 05 — if that notebook's
logic ever changes, update both and re-run `pytest tests/test_features.py`.

## Definition of done, checked off

- [x] Structured repo with config files, requirements, and this README
- [x] Inference pipeline in Python modules loading saved fitted objects + registered model
- [x] Data validation (Great Expectations) + versioning (DVC) + tracking (MLflow)
- [x] Unit and integration tests passing with one command (`pytest tests/`)
- [x] FastAPI service with health, model info, and predict routes
- [x] Docker Compose bringing up the database, the API, and artifact storage together
- [x] CI/CD pipeline running tests and building the image
- [x] Logging and monitoring in place, with prediction logs stored
