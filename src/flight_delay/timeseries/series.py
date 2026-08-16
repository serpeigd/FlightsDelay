"""Daily delay-rate series and the calendar features that explain most of it.

The classification model answers "is *this flight* going to be late". This one
answers a different question that an operations team actually asks: "how bad is
tomorrow going to be at this airport". The unit is a day, not a flight, and the
useful horizon is days rather than hours.

Holidays are included because US air traffic is visibly shaped by them, and the
proximity to one matters as much as the day itself: the Wednesday before
Thanksgiving is not a holiday and is one of the worst travel days of the year.
``pandas`` ships the federal calendar, so this costs no new dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from flight_delay.config import Paths

if TYPE_CHECKING:
    import duckdb

#: A daily rate computed from a handful of flights is noise, not a series.
MIN_FLIGHTS_PER_DAY: Final = 30

#: Weekly seasonality is the dominant cycle in this data.
SEASON: Final = 7


def national_series(con: duckdb.DuckDBPyConnection, paths: Paths) -> pd.DataFrame:
    """One row per calendar day: how many flights, and what share arrived late."""
    src = f"read_parquet('{paths.table('model_table')}/**/*.parquet', hive_partitioning=true)"
    frame = con.execute(
        f"""
        SELECT flight_date AS date,
               count(*)    AS flights,
               avg(label)  AS rate
        FROM {src}
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def airport_series(
    con: duckdb.DuckDBPyConnection,
    paths: Paths,
    *,
    top_n: int = 10,
    min_flights: int = MIN_FLIGHTS_PER_DAY,
) -> pd.DataFrame:
    """Daily series for the busiest origins, one row per airport-day."""
    src = f"read_parquet('{paths.table('model_table')}/**/*.parquet', hive_partitioning=true)"
    frame = con.execute(
        f"""
        WITH busiest AS (
            SELECT origin FROM {src}
            GROUP BY 1 ORDER BY count(*) DESC LIMIT {top_n}
        )
        SELECT origin, flight_date AS date, count(*) AS flights, avg(label) AS rate
        FROM {src}
        WHERE origin IN (SELECT origin FROM busiest)
        GROUP BY 1, 2
        HAVING count(*) >= {min_flights}
        ORDER BY 1, 2
        """
    ).df()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def holiday_dates(start: str = "2022-12-01", end: str = "2025-01-31") -> pd.DatetimeIndex:
    """Padded beyond the data so distance-to-holiday is defined at the edges."""
    holidays = USFederalHolidayCalendar().holidays(start=pd.Timestamp(start), end=pd.Timestamp(end))
    return pd.DatetimeIndex(holidays)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week, month, and distance to the nearest federal holiday."""
    out = frame.copy()
    dates = out["date"]

    out["day_of_week"] = dates.dt.dayofweek
    out["month"] = dates.dt.month
    out["day_of_year"] = dates.dt.dayofyear
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    holidays = holiday_dates().to_numpy(dtype="datetime64[D]")
    days = dates.to_numpy(dtype="datetime64[D]")
    # Signed distance to the nearest holiday: negative before, positive after.
    # The Wednesday before Thanksgiving is not a holiday and is one of the
    # busiest travel days of the year, so the offset carries the signal that
    # a plain is_holiday flag misses.
    deltas = (days[:, None] - holidays[None, :]).astype("timedelta64[D]").astype(int)
    nearest = np.argmin(np.abs(deltas), axis=1)
    out["days_from_holiday"] = deltas[np.arange(len(days)), nearest]
    out["is_holiday"] = (out["days_from_holiday"] == 0).astype(int)
    out["near_holiday"] = (out["days_from_holiday"].abs() <= 3).astype(int)

    return out


CALENDAR_FEATURES: Final[tuple[str, ...]] = (
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "days_from_holiday",
    "is_holiday",
    "near_holiday",
)


def add_lags(
    frame: pd.DataFrame,
    *,
    horizon: int,
    column: str = "rate",
    lags: tuple[int, ...] = (1, 2, 3, 7, 14, 28),
    rolling: tuple[int, ...] = (7, 28),
    group: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Lagged and rolling values that exist ``horizon`` days before the target.

    Everything here is built *for a specific horizon*, which is the only way to
    keep it honest. Forecasting a week ahead, the most recent observation
    available is from seven days before the target day, so:

    - only lags of ``horizon`` or more are produced; ``lag_1`` in a 7-day model
      is a value nobody has yet;
    - rolling windows are shifted by ``horizon``, not by one. A 7-day mean
      shifted by a single day contains six days the forecaster cannot see. That
      shift is the most common way a forecasting benchmark quietly turns into a
      lookup, and it does not announce itself -- it just makes the model look
      good.

    Returns the frame and the feature names, so the caller cannot accidentally
    select a column the horizon forbids.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")

    out = frame.copy()
    values = out.groupby(group)[column] if group else out[column]
    names: list[str] = []

    for lag in lags:
        if lag < horizon:
            continue
        out[f"lag_{lag}"] = values.shift(lag)
        names.append(f"lag_{lag}")

    shifted = out.groupby(group)[column].shift(horizon) if group else out[column].shift(horizon)
    for window in rolling:
        name = f"roll_mean_{window}"
        if group:
            rolled = (
                shifted.groupby(out[group])
                .rolling(window, min_periods=window)
                .mean()
                .reset_index(level=0, drop=True)
            )
        else:
            rolled = shifted.rolling(window, min_periods=window).mean()
        out[name] = rolled
        names.append(name)

    return out, [*CALENDAR_FEATURES, *names]
