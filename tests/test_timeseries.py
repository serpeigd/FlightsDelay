from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_delay.timeseries.backtest import score, seasonal_naive_scale
from flight_delay.timeseries.series import (
    CALENDAR_FEATURES,
    add_calendar_features,
    add_lags,
)


def daily(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "rate": np.linspace(0.1, 0.4, n)})


def test_calendar_features_are_added() -> None:
    out = add_calendar_features(daily())
    for column in CALENDAR_FEATURES:
        assert column in out.columns


def test_thanksgiving_is_flagged_and_the_wednesday_before_is_not() -> None:
    """The day before Thanksgiving is one of the busiest travel days of the
    year and is not itself a holiday, which is why the signed distance exists
    rather than only a boolean."""
    frame = pd.DataFrame({"date": pd.to_datetime(["2023-11-22", "2023-11-23"])})
    out = add_calendar_features(frame)

    assert out.loc[1, "is_holiday"] == 1  # Thanksgiving
    assert out.loc[0, "is_holiday"] == 0  # the Wednesday before
    assert out.loc[0, "days_from_holiday"] == -1
    assert out.loc[0, "near_holiday"] == 1


def test_lags_shorter_than_the_horizon_are_never_produced() -> None:
    """lag_1 in a 7-day-ahead model is a value nobody has yet."""
    _, features = add_lags(add_calendar_features(daily()), horizon=7)
    assert "lag_1" not in features
    assert "lag_2" not in features
    assert "lag_3" not in features
    assert "lag_7" in features
    assert "lag_14" in features


def test_horizon_one_keeps_the_short_lags() -> None:
    _, features = add_lags(add_calendar_features(daily()), horizon=1)
    assert "lag_1" in features


def test_lag_values_come_from_the_right_row() -> None:
    frame = add_calendar_features(daily(30))
    out, _ = add_lags(frame, horizon=1, lags=(1, 7), rolling=())
    assert out.loc[10, "lag_1"] == pytest.approx(frame.loc[9, "rate"])
    assert out.loc[10, "lag_7"] == pytest.approx(frame.loc[3, "rate"])


def test_rolling_window_is_shifted_by_the_horizon_not_by_one() -> None:
    """A 7-day mean shifted a single day contains six days a 7-day-ahead
    forecaster cannot see. This is the leak that makes a benchmark look good
    without announcing itself."""
    frame = add_calendar_features(daily(60))
    out, _ = add_lags(frame, horizon=7, lags=(), rolling=(7,))

    row = 40
    expected = frame.loc[row - 13 : row - 7, "rate"].mean()
    assert out.loc[row, "roll_mean_7"] == pytest.approx(expected)

    # Sanity: the naive one-day shift would have used later data.
    wrong = frame.loc[row - 7 : row - 1, "rate"].mean()
    assert out.loc[row, "roll_mean_7"] != pytest.approx(wrong)


def test_rolling_window_never_contains_the_target_day() -> None:
    frame = add_calendar_features(daily(60))
    rates = frame["rate"].to_numpy()
    for horizon in (1, 3, 7):
        out, _ = add_lags(frame, horizon=horizon, lags=(), rolling=(7,))
        row = 40
        # The series is increasing, so the mean of an admissible window can
        # never exceed the most recent value the forecaster is allowed to see.
        latest_allowed = rates[row - horizon]
        window_mean = out["roll_mean_7"].to_numpy()[row]
        assert window_mean <= latest_allowed + 1e-12


def test_horizon_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="horizon"):
        add_lags(add_calendar_features(daily()), horizon=0)


def test_mase_scale_is_the_seasonal_difference() -> None:
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4])
    # Week-over-week change is +0.1 for each of the three overlapping pairs.
    assert seasonal_naive_scale(y, season=7) == pytest.approx(0.1)


def test_mase_below_one_means_the_baseline_was_beaten() -> None:
    y = np.array([0.2, 0.3, 0.4])
    good = score("m", 1, y, y + 0.01, scale=0.1)
    bad = score("m", 1, y, y + 0.5, scale=0.1)

    assert good.mase == pytest.approx(0.1)
    assert good.beats_seasonal_naive
    assert not bad.beats_seasonal_naive


def test_a_series_shorter_than_a_season_cannot_be_scaled() -> None:
    with pytest.raises(ValueError, match="shorter than one season"):
        seasonal_naive_scale(np.arange(5, dtype=float))


def test_a_flat_series_has_no_defined_mase() -> None:
    with pytest.raises(ValueError, match="undefined"):
        seasonal_naive_scale(np.full(30, 0.2))
