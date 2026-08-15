"""Tests for the leakage contract.

These are the most important tests in the repo: a model that reads an
``ARRIVED`` column scores beautifully and means nothing.
"""

from __future__ import annotations

import re

from flight_delay.ingest.schema import (
    BY_NAME,
    COLUMNS,
    KEPT_COLUMNS,
    PARTITION_BY,
    SCENARIO_A_COLUMNS,
    SCENARIO_B_COLUMNS,
    Availability,
    columns_available_at,
)

#: Columns that are recorded only once the outcome is known. If any of these
#: reaches a feature set, the score is meaningless.
POST_HOC = (
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrTime",
    "ArrivalDelayGroups",
    "ActualElapsedTime",
    "AirTime",
    "WheelsOn",
    "TaxiIn",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
    "Cancelled",
    "Diverted",
    "CancellationCode",
)


def test_column_names_are_unique() -> None:
    assert len(BY_NAME) == len(COLUMNS)


def test_exactly_one_label() -> None:
    labels = [c.name for c in COLUMNS if c.availability is Availability.LABEL]
    assert labels == ["ArrDel15"]


def test_post_hoc_columns_are_marked_arrived() -> None:
    for name in POST_HOC:
        assert BY_NAME[name].availability is Availability.ARRIVED, name


def test_scenario_a_excludes_everything_known_only_after_departure() -> None:
    forbidden = {"DepDelay", "DepDelayMinutes", "DepDel15", "TaxiOut", "WheelsOff", "DepTime"}
    assert forbidden.isdisjoint(SCENARIO_A_COLUMNS)
    assert set(POST_HOC).isdisjoint(SCENARIO_A_COLUMNS)


def test_scenario_b_adds_departure_but_still_excludes_the_outcome() -> None:
    assert set(SCENARIO_A_COLUMNS) < set(SCENARIO_B_COLUMNS)
    assert "DepDelay" in SCENARIO_B_COLUMNS
    assert set(POST_HOC).isdisjoint(SCENARIO_B_COLUMNS)


def test_no_scenario_contains_the_label() -> None:
    assert "ArrDel15" not in SCENARIO_A_COLUMNS
    assert "ArrDel15" not in SCENARIO_B_COLUMNS


def test_diverted_leg_columns_were_dropped() -> None:
    """The 45 ``Div1*``-``Div5*`` columns describe diverted-flight legs and are
    almost entirely null. ``Diverted`` itself is kept and also starts with
    "Div", so the prefix alone is not a safe filter."""
    assert not [name for name in KEPT_COLUMNS if re.fullmatch(r"Div[1-5].*", name)]
    assert "Diverted" in KEPT_COLUMNS


def test_partition_keys_exist_and_are_scheduled() -> None:
    for key in PARTITION_BY:
        assert key in BY_NAME
        assert BY_NAME[key].availability is Availability.SCHEDULED


def test_clock_columns_stay_strings() -> None:
    """'0800' parsed as an integer loses its padding, and '2400' is a real
    value in this feed; both are converted deliberately downstream."""
    for name in ("CRSDepTime", "CRSArrTime", "DepTime", "ArrTime", "WheelsOff", "WheelsOn"):
        assert BY_NAME[name].sql_type == "string", name


def test_availability_query_partitions_the_schema() -> None:
    total = sum(
        len(columns_available_at(a))
        for a in (
            Availability.SCHEDULED,
            Availability.DEPARTED,
            Availability.ARRIVED,
            Availability.LABEL,
        )
    )
    assert total == len(COLUMNS)
