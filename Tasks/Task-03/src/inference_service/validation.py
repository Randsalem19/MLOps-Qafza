"""Input validation.

Design decision (documented, not accidental): Great Expectations 1.x builds a
metrics graph on every ``batch.validate()`` call, which costs ~150-250ms even
on a single row and even when the Data Source / Asset / Suite are built once
and reused. That is fine for a nightly data-quality job but too slow for a
p50-sensitive single-order prediction endpoint.

So this service validates on two tracks, both built from the SAME
``config/expectations.json`` so they can never drift apart:

1. ``validate_fast`` — plain pandas checks (not-null, numeric ranges,
   allowed categories). Used on every request, including single-order
   ``/predict``. Sub-millisecond.
2. ``validate_with_great_expectations`` — a real GX Expectation Suite built
   from the same expectations file. Used on ``/predict/batch`` (the cost is
   amortized across many rows) and available as a standalone check for
   offline / scheduled data-quality runs (see scripts/run_ge_validation.py).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import pandas as pd

from .config import get_expectations


@dataclass
class ValidationResult:
    is_valid: bool
    failures: list[str] = field(default_factory=list)


def validate_fast(frame: pd.DataFrame) -> ValidationResult:
    expectations = get_expectations()
    failures: list[str] = []

    for column in expectations["not_null"]:
        if column not in frame.columns or frame[column].isna().any():
            failures.append(f"'{column}' must not be null")

    for column in expectations.get("not_blank", []):
        if column in frame.columns and (frame[column].astype(str).str.strip() == "").any():
            failures.append(f"'{column}' must not be blank")

    for column, bounds in expectations["numeric_ranges"].items():
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        out_of_range = series.isna() | (series < bounds["min"]) | (series > bounds["max"])
        if out_of_range.any():
            failures.append(f"'{column}' must be between {bounds['min']} and {bounds['max']}")

    for column, allowed in expectations["categorical_sets"].items():
        if column not in frame.columns:
            continue
        bad_values = set(frame[column].dropna().unique()) - set(allowed)
        if bad_values:
            failures.append(f"'{column}' contains values outside the allowed set: {sorted(bad_values)}")

    return ValidationResult(is_valid=not failures, failures=failures)


@functools.lru_cache(maxsize=1)
def _build_ge_suite():
    """Built once per process. Returns (batch_definition, suite) so callers
    can call ``batch_definition.get_batch(...).validate(suite)`` repeatedly."""
    import great_expectations as gx

    expectations = get_expectations()
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("inference_service_pandas")
    data_asset = data_source.add_dataframe_asset(name="orders")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("orders_batch")

    suite = context.suites.add(gx.ExpectationSuite(name="order_input_suite"))
    for column in expectations["not_null"]:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=column))
    for column, bounds in expectations["numeric_ranges"].items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column, min_value=bounds["min"], max_value=bounds["max"]
            )
        )
    for column, allowed in expectations["categorical_sets"].items():
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column=column, value_set=allowed))
    return batch_definition, suite


def validate_with_great_expectations(frame: pd.DataFrame) -> ValidationResult:
    batch_definition, suite = _build_ge_suite()
    batch = batch_definition.get_batch(batch_parameters={"dataframe": frame})
    result = batch.validate(suite)
    failures = [
        r["expectation_config"]["kwargs"].get("column", "?") for r in result.results if not r["success"]
    ]
    return ValidationResult(
        is_valid=bool(result.success), failures=[f"GE check failed for '{c}'" for c in failures]
    )
