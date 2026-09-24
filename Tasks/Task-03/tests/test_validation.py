import pandas as pd

from inference_service.validation import validate_fast, validate_with_great_expectations


def test_valid_order_passes_fast_validation(sample_order):
    frame = pd.DataFrame([sample_order])
    result = validate_fast(frame)
    assert result.is_valid
    assert result.failures == []


def test_unknown_state_fails_fast_validation(make_order):
    frame = pd.DataFrame([make_order(customer_state="ZZ")])
    result = validate_fast(frame)
    assert not result.is_valid
    assert any("customer_state" in failure for failure in result.failures)


def test_negative_price_fails_fast_validation(make_order):
    frame = pd.DataFrame([make_order(price_total=-10.0)])
    result = validate_fast(frame)
    assert not result.is_valid
    assert any("price_total" in failure for failure in result.failures)


def test_null_required_field_fails_fast_validation(make_order):
    frame = pd.DataFrame([make_order(customer_state=None)])
    result = validate_fast(frame)
    assert not result.is_valid


def test_great_expectations_suite_agrees_on_a_valid_order(sample_order):
    frame = pd.DataFrame([sample_order])
    result = validate_with_great_expectations(frame)
    assert result.is_valid


def test_great_expectations_suite_agrees_on_an_invalid_order(make_order):
    frame = pd.DataFrame([make_order(customer_state="ZZ")])
    result = validate_with_great_expectations(frame)
    assert not result.is_valid
