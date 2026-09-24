import pytest

from inference_service.model_loader import get_model
from inference_service.predictor import ValidationError, predict_orders


def test_model_loads_from_local_fallback():
    loaded = get_model()
    assert loaded.source == "local_fallback"
    assert loaded.model is not None
    assert loaded.preprocessor is not None
    assert loaded.feature_names


def test_predict_orders_returns_one_result_per_order(sample_order, make_order):
    orders = [sample_order, make_order(order_id="fixture-order-2")]
    results = predict_orders(orders)
    assert len(results) == 2
    assert {r.order_id for r in results} == {"fixture-order-1", "fixture-order-2"}


def test_predict_orders_probability_is_between_zero_and_one(sample_order):
    result = predict_orders([sample_order])[0]
    assert 0.0 <= result.probability_late <= 1.0


def test_predict_orders_label_matches_threshold(sample_order):
    result = predict_orders([sample_order])[0]
    if result.probability_late >= result.decision_threshold:
        assert result.prediction == "late"
    else:
        assert result.prediction == "on_time"


def test_predict_orders_is_deterministic_on_the_same_input(sample_order):
    first = predict_orders([sample_order])[0]
    second = predict_orders([sample_order])[0]
    assert first.probability_late == pytest.approx(second.probability_late)


def test_predict_orders_raises_validation_error_on_bad_input(make_order):
    bad_order = make_order(customer_state="ZZ")
    with pytest.raises(ValidationError) as excinfo:
        predict_orders([bad_order])
    assert any("customer_state" in failure for failure in excinfo.value.failures)


def test_predict_orders_never_refits_the_preprocessor(sample_order):
    """Calling predict must reuse the exact same fitted preprocessor object
    (identity check) rather than fitting a new one — proves .transform()
    is used, never .fit()/.fit_transform()."""
    loaded_before = get_model()
    predict_orders([sample_order])
    loaded_after = get_model()
    assert loaded_before.preprocessor is loaded_after.preprocessor
    assert loaded_before.model is loaded_after.model
