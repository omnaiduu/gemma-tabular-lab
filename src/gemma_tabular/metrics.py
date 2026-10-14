"""Numerical metrics without dependency on the training environment."""

import math


def percentile(values, probability):
    values = sorted(values)
    if not values:
        raise ValueError("No observations")
    index = probability * (len(values) - 1)
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def price_metrics(actual, predicted):
    actual, predicted = list(actual), list(predicted)
    if not actual or len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must be non-empty and aligned")
    if any(not math.isfinite(v) or v < 0 for v in actual + predicted):
        raise ValueError("Price metrics require finite non-negative values")
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    log_errors = [(math.log1p(a) - math.log1p(p)) ** 2 for a, p in zip(actual, predicted)]
    positive = [(a, p) for a, p in zip(actual, predicted) if a > 0]
    return {
        "n": len(actual),
        "mae": sum(errors) / len(errors),
        "rmsle": math.sqrt(sum(log_errors) / len(log_errors)),
        "median_absolute_error": percentile(errors, 0.5),
        "p90_absolute_error": percentile(errors, 0.9),
        "factor_two_error_rate": sum(p > 2*a or p < a/2 for a, p in positive) / len(positive) if positive else None,
    }
