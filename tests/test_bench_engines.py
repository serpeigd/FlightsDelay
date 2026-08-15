from __future__ import annotations

import re

import pytest

from flight_delay.bench.engines import WORKLOADS, months_filter


def test_one_month_selects_only_january_2023() -> None:
    assert months_filter(1).endswith("<= 202301")


def test_full_range_ends_at_december_2024() -> None:
    assert months_filter(24).endswith("<= 202412")


def test_month_boundaries_do_not_wrap_across_the_year() -> None:
    """Year*100+Month keeps December 2023 below January 2024; a naive
    year+month sum would collide."""
    assert months_filter(12).endswith("<= 202312")
    assert months_filter(13).endswith("<= 202401")


@pytest.mark.parametrize("n", [0, 25, -1])
def test_out_of_range_scales_are_rejected(n: int) -> None:
    with pytest.raises(ValueError, match="n_months"):
        months_filter(n)


@pytest.mark.parametrize("name", list(WORKLOADS))
def test_workloads_are_parameterised_for_both_engines(name: str) -> None:
    sql = WORKLOADS[name].format(table="t", months=months_filter(6))
    assert "{" not in sql
    assert "t" in sql


def test_window_ordering_is_total() -> None:
    """4,819 rows share a Tail_Number, FlightDate and CRSDepTime. Without
    tiebreakers, lag() is non-deterministic and the engines disagreed on real
    output -- and this expression is a model feature, not just a benchmark."""
    sql = WORKLOADS["inbound_leg_delay"]
    order_by = re.search(r"ORDER BY(.+?)\n\s*\)", sql, re.DOTALL)
    assert order_by is not None
    keys = order_by.group(1)
    for column in ("FlightDate", "CRSDepTime", "Flight_Number", "Origin", "Dest"):
        assert column in keys, column


def test_identifiers_are_double_quoted_not_backticked() -> None:
    """Backticks would not parse on DuckDB. Double quotes work on both, given
    spark.sql.ansi.doubleQuotedIdentifiers."""
    for sql in WORKLOADS.values():
        assert "`" not in sql
