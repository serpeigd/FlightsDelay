"""Turn an API request into the exact frame the model was trained on.

This is where training-serving skew is born, so the derivations live in one
place and are tested against the SQL that produced the training table.

Three fields cannot be supplied by a caller and are looked up or left missing:

- ``origin_prior_rate`` / ``carrier_prior_rate`` / ``origin_prior_flights`` come
  from a small reference table exported alongside the model. At training time
  they are the *previous* calendar month's rate; at serving time the most
  recent month available plays that role.
- ``inbound_delay`` and its companions describe the aircraft's previous leg.
  A caller may know them; most will not. Missing is a legitimate answer and the
  model was trained on a majority of exactly that case -- 72.8% of training
  rows have no usable inbound -- so a null here is in-distribution rather than
  a degraded prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from flight_delay.features.build import CATEGORICAL, SCENARIO_A_FEATURES


@dataclass(frozen=True, slots=True)
class PriorRates:
    """Most recent monthly delay rate per origin and per carrier."""

    origin_rate: dict[str, float]
    origin_flights: dict[str, int]
    carrier_rate: dict[str, float]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> PriorRates:
        """Read a long-format table: one row per (kind, key).

        Long rather than wide because there are 355 airports and 15 carriers;
        a wide table would be mostly padding, and joining two unrelated
        dimensions into one row invites exactly that.
        """
        origins = frame[frame["kind"] == "origin"]
        carriers = frame[frame["kind"] == "carrier"]
        return cls(
            origin_rate=dict(zip(origins["key"], origins["rate"], strict=True)),
            origin_flights=dict(zip(origins["key"], origins["flights"], strict=True)),
            carrier_rate=dict(zip(carriers["key"], carriers["rate"], strict=True)),
        )


def clock_to_minutes(clock: str) -> int:
    """``'0830'`` to minutes past midnight.

    ``'2400'`` is a real value in this feed and is left at 1440 rather than
    wrapped to 0, matching what the training SQL did.
    """
    if len(clock) != 4 or not clock.isdigit():
        raise ValueError(f"expected a 4-digit clock like '0830', got {clock!r}")
    hours, minutes = int(clock[:2]), int(clock[2:])
    if hours > 24 or minutes > 59:
        raise ValueError(f"not a valid clock time: {clock!r}")
    return hours * 60 + minutes


def build_row(
    *,
    flight_date: date,
    carrier: str,
    origin: str,
    dest: str,
    scheduled_departure: str,
    scheduled_arrival: str,
    distance: float,
    scheduled_elapsed_minutes: float,
    priors: PriorRates,
    inbound_delay_minutes: float | None = None,
    inbound_turnaround_minutes: float | None = None,
) -> pd.DataFrame:
    """One row with exactly the training columns, in the training order."""
    dep_minutes = clock_to_minutes(scheduled_departure)
    # BTS numbers weekdays 1-7 from Monday; date.isoweekday() agrees.
    day_of_week = flight_date.isoweekday()

    values: dict[str, object] = {
        "dep_minute_of_day": dep_minutes,
        "arr_minute_of_day": clock_to_minutes(scheduled_arrival),
        "day_of_week": day_of_week,
        "month": flight_date.month,
        "distance": distance,
        "crs_elapsed": scheduled_elapsed_minutes,
        "carrier": carrier,
        "origin": origin,
        "dest": dest,
        "origin_prior_rate": priors.origin_rate.get(origin),
        "carrier_prior_rate": priors.carrier_rate.get(carrier),
        # Null when the caller does not know the inbound leg. That is the
        # common case and the model was trained on it; there is no separate
        # flag because the null carries the same information.
        "inbound_delay": inbound_delay_minutes,
        "inbound_turnaround_minutes": inbound_turnaround_minutes,
    }

    missing = set(SCENARIO_A_FEATURES) - set(values)
    extra = set(values) - set(SCENARIO_A_FEATURES)
    if missing or extra:
        raise RuntimeError(
            f"serving features drifted from training: missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )

    frame = pd.DataFrame([values])[list(SCENARIO_A_FEATURES)]
    for column in CATEGORICAL:
        frame[column] = frame[column].astype("category")
    return frame
