import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Imported inside the fixture so the _isolated_project autouse fixture
    # (which sets PROJECT_ROOT/CONFIG_PATH) has already run before app.main
    # builds its module-level `config = get_config()`.
    from app.main import app

    return TestClient(app)


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_returns_metadata(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "order-delivery-classifier"
    assert body["feature_count"] > 0
    assert body["primary_metric"] == "PR-AUC"


def test_predict_returns_a_valid_prediction(client, sample_order):
    response = client.post("/predict", json=sample_order)
    assert response.status_code == 200
    body = response.json()
    assert body["order_id"] == sample_order["order_id"]
    assert body["prediction"] in ("late", "on_time")
    assert 0.0 <= body["probability_late"] <= 1.0


def test_predict_rejects_bad_payload_clearly(client, make_order):
    bad_order = make_order(customer_state="ZZ")
    response = client.post("/predict", json=bad_order)
    assert response.status_code == 422
    body = response.json()
    assert "failures" in body
    assert any("customer_state" in failure for failure in body["failures"])


def test_predict_rejects_missing_required_field(client, sample_order):
    incomplete = dict(sample_order)
    del incomplete["price_total"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422  # pydantic schema validation, not our custom validator


def test_predict_batch_scores_every_order(client, sample_order, make_order):
    payload = {"orders": [sample_order, make_order(order_id="fixture-order-2")]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert {p["order_id"] for p in predictions} == {"fixture-order-1", "fixture-order-2"}


def test_predict_batch_rejects_when_any_order_is_invalid(client, sample_order, make_order):
    payload = {"orders": [sample_order, make_order(order_id="bad", price_total=-5.0)]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(client, sample_order):
    client.post("/predict", json=sample_order)  # generate at least one data point
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predictions_total" in response.text
    assert "prediction_latency_seconds" in response.text


def test_docs_are_reachable(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/predict" in schema["paths"]
    assert "/predict/batch" in schema["paths"]
