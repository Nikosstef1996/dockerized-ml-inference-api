"""Simple forecasting baselines. Replace with a trained ML model on Day 2."""


def seasonal_naive_next_day(last_24_hours_mwh: list[float]) -> list[float]:
    """Forecast each of the next 24 hours using the same hour of the previous day.

    This is a baseline, not a trained ML model.
    """
    return list(last_24_hours_mwh)
