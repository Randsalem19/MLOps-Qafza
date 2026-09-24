import pandas as pd
import pytest

from inference_service.features import build_feature_frame, orders_to_dataframe, select_model_columns


def test_build_feature_frame_derives_expected_columns(sample_order):
    frame = orders_to_dataframe([sample_order])
    engineered = build_feature_frame(frame)

    assert engineered.loc[0, "purchase_month"] == 5
    assert engineered.loc[0, "purchase_hour"] == 10
    assert engineered.loc[0, "purchase_is_weekend"] == 0  # 2018-05-14 is a Monday
    # 2018-05-29T00:00:00 minus 2018-05-14T10:32:00, as fractional days (matches notebook 05 exactly)
    assert engineered.loc[0, "estimated_delivery_days"] == pytest.approx(14.5611, abs=1e-3)


def test_build_feature_frame_flags_weekend_purchase(make_order):
    # 2018-05-19 is a Saturday
    order = make_order(order_purchase_timestamp="2018-05-19T09:00:00")
    engineered = build_feature_frame(orders_to_dataframe([order]))
    assert engineered.loc[0, "purchase_is_weekend"] == 1


def test_select_model_columns_returns_exact_training_schema(sample_order):
    engineered = build_feature_frame(orders_to_dataframe([sample_order]))
    model_frame = select_model_columns(engineered)

    assert "order_id" not in model_frame.columns
    assert "customer_state" in model_frame.columns
    assert "purchase_month" in model_frame.columns


def test_select_model_columns_raises_on_missing_column(sample_order):
    frame = pd.DataFrame([sample_order]).drop(columns=["price_total"])
    engineered = build_feature_frame(frame)
    with pytest.raises(ValueError, match="price_total"):
        select_model_columns(engineered)
