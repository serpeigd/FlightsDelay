from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from flight_delay.features.build import CATEGORICAL, SCENARIO_A_FEATURES
from flight_delay.serving.features import PriorRates, build_row, clock_to_minutes

PRIORS = PriorRates(
    origin_rate={"JFK": 0.24},
    origin_flights={"JFK": 30_000},
    carrier_rate={"AA": 0.22},
)


def row(**overrides: object) -> pd.DataFrame:
    kwargs: dict[str, object] = {
        "flight_date": date(2024, 7, 15),  # a Monday
        "carrier": "AA",
        "origin": "JFK",
        "dest": "LAX",
        "scheduled_departure": "0830",
        "scheduled_arrival": "1145",
        "distance": 2475.0,
        "scheduled_elapsed_minutes": 375.0,
        "priors": PRIORS,
    }
    kwargs.update(overrides)
    return build_row(**kwargs)  # type: ignore[arg-type]


def test_clock_parsing_matches_the_training_sql() -> None:
    assert clock_to_minutes("0000") == 0
    assert clock_to_minutes("0830") == 510
    assert clock_to_minutes("2359") == 1439


def test_2400_is_not_wrapped_to_zero() -> None:
    """'2400' is a real value in this feed and the training SQL left it at
    1440. Wrapping it here would place a midnight flight at the start of the
    day instead of the end."""
    assert clock_to_minutes("2400") == 1440


@pytest.mark.parametrize("bad", ["830", "08:30", "0870", "2500", "abcd", ""])
def test_malformed_clocks_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="clock"):
        clock_to_minutes(bad)


def test_serving_columns_match_training_exactly_and_in_order() -> None:
    """The whole point of this module. A column added to the training schema
    and forgotten here would silently shift the model's inputs."""
    assert list(row().columns) == list(SCENARIO_A_FEATURES)


def test_categoricals_keep_their_dtype() -> None:
    frame = row()
    for column in CATEGORICAL:
        assert isinstance(frame[column].dtype, pd.CategoricalDtype), column


def test_dep_hour_is_integer_division_as_in_the_sql() -> None:
    """The SQL computed dep_minute_of_day / 60 with integer semantics. A float
    here would fall on the other side of split points the trees learned."""
    frame = row(scheduled_departure="0859")
    assert frame["dep_minute_of_day"].iloc[0] == 539
    assert frame["dep_hour"].iloc[0] == 8


def test_weekday_numbering_matches_bts() -> None:
    """BTS numbers Monday as 1; date.isoweekday() agrees. Using weekday(),
    which starts at 0, would shift every day by one."""
    monday = row(flight_date=date(2024, 7, 15))
    sunday = row(flight_date=date(2024, 7, 21))
    assert monday["day_of_week"].iloc[0] == 1
    assert monday["is_weekend"].iloc[0] == 0
    assert sunday["day_of_week"].iloc[0] == 7
    assert sunday["is_weekend"].iloc[0] == 1


def test_saturday_and_sunday_are_the_weekend() -> None:
    for day, expected in ((date(2024, 7, 19), 0), (date(2024, 7, 20), 1)):
        assert row(flight_date=day)["is_weekend"].iloc[0] == expected


def test_priors_are_looked_up_not_supplied_by_the_caller() -> None:
    frame = row()
    assert frame["origin_prior_rate"].iloc[0] == pytest.approx(0.24)
    assert frame["carrier_prior_rate"].iloc[0] == pytest.approx(0.22)


def test_an_unknown_airport_yields_a_null_prior_rather_than_an_error() -> None:
    """Gradient boosting handles NaN natively, and a route the reference table
    has never seen is a normal request, not a failure."""
    frame = row(origin="ZZZ")
    assert pd.isna(frame["origin_prior_rate"].iloc[0])
    assert pd.isna(frame["origin_prior_flights"].iloc[0])


def test_a_missing_inbound_is_in_distribution() -> None:
    """72.8% of training rows had no usable inbound, so null here is the
    common case rather than a degraded prediction."""
    frame = row()
    assert pd.isna(frame["inbound_delay"].iloc[0])
    assert frame["inbound_known"].iloc[0] == 0


def test_a_known_inbound_sets_the_flag() -> None:
    frame = row(inbound_delay_minutes=42.0, inbound_turnaround_minutes=180.0)
    assert frame["inbound_delay"].iloc[0] == 42.0
    assert frame["inbound_known"].iloc[0] == 1
    assert frame["inbound_turnaround_minutes"].iloc[0] == 180.0


def test_priors_load_from_a_long_format_frame() -> None:
    priors = PriorRates.from_frame(
        pd.DataFrame(
            {
                "kind": ["origin", "origin", "carrier"],
                "key": ["JFK", "LAX", "AA"],
                "rate": [0.3, 0.28, 0.2],
                "flights": [10, 20, 30],
            }
        )
    )
    assert priors.origin_rate == pytest.approx({"JFK": 0.3, "LAX": 0.28})
    assert priors.origin_flights == {"JFK": 10, "LAX": 20}
    assert priors.carrier_rate == pytest.approx({"AA": 0.2})


def test_carriers_and_origins_do_not_bleed_into_each_other() -> None:
    """An airport code and a carrier code can collide -- 'AS' is Alaska
    Airlines, and three-letter codes share a namespace with nothing here, but
    a wide table joined on nothing would have mixed them."""
    priors = PriorRates.from_frame(
        pd.DataFrame(
            {
                "kind": ["origin", "carrier"],
                "key": ["AS", "AS"],
                "rate": [0.11, 0.99],
                "flights": [1, 2],
            }
        )
    )
    assert priors.origin_rate["AS"] == pytest.approx(0.11)
    assert priors.carrier_rate["AS"] == pytest.approx(0.99)
