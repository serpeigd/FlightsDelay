from __future__ import annotations

import duckdb
import pytest

from flight_delay.bench.scale import (
    ARCHIVE_URL,
    Projection,
    memory_verdict,
    probe_feed,
    rows_per_scale,
)


def test_a_month_that_is_not_published_is_absent_not_an_error() -> None:
    """The PREZIP directory does not carry every month, and one 404 must not
    lose the other four hundred measurements."""
    sizes = {ARCHIVE_URL.format(year=2023, month=m): 1_000 for m in range(1, 13)}
    probe = probe_feed([2023, 2024], head=lambda url: sizes.get(url, 0))

    assert probe.probed == 24
    assert probe.found == 12
    assert probe.compressed_bytes == 12_000
    assert probe.first_month == (2023, 1)
    assert probe.last_month == (2023, 12)
    assert probe.absent_years == (2024,)


def test_nothing_answering_is_reported_rather_than_guessed() -> None:
    probe = probe_feed([2023], head=lambda _: 0)
    assert probe.found == 0
    assert probe.compressed_bytes == 0
    assert probe.first_month is None


def test_the_projection_scales_measured_rows_by_measured_bytes() -> None:
    """Two years of a feed measured exactly, projected onto ten times the
    compressed bytes."""
    projection = Projection(
        rows_measured=1_000_000,
        compressed_bytes_measured=100_000_000,
        parquet_bytes_measured=20_000_000,
        feed_compressed_bytes=1_000_000_000,
    )
    assert projection.parquet_bytes_per_row == pytest.approx(20.0)
    assert projection.estimated_feed_rows == 10_000_000
    assert projection.estimated_feed_parquet_bytes == 200_000_000
    assert projection.multiple_of_measured == pytest.approx(10.0)


def test_the_memory_verdict_states_its_assumption_and_the_row_count() -> None:
    projection = Projection(
        rows_measured=1_000_000,
        compressed_bytes_measured=100_000_000,
        parquet_bytes_measured=20_000_000,
        feed_compressed_bytes=1_000_000_000,
    )
    verdict = memory_verdict(projection, machine_bytes=1_000_000_000, expansion=3.0)

    # 200 MB of Parquet x3 = 600 MB against a 1 GB machine.
    assert verdict["estimated_working_set_bytes"] == 600_000_000
    assert verdict["fits"] is True
    # 1 GB / (20 bytes x 3) rows.
    assert verdict["rows_that_fit"] == pytest.approx(16_666_667, rel=1e-6)

    tighter = memory_verdict(projection, machine_bytes=100_000_000, expansion=3.0)
    assert tighter["fits"] is False


def test_rows_per_scale_counts_rather_than_divides() -> None:
    """Months are not uniform, so a scale's row count is a query, not a
    twenty-fourth of the total."""
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE flights AS
        SELECT * FROM (VALUES
            (2023, 1), (2023, 1), (2023, 1),
            (2023, 2),
            (2023, 3), (2023, 3)
        ) AS t("Year", "Month")
        """
    )
    counts = rows_per_scale(con, "flights", (1, 2, 3))
    assert counts == {1: 3, 2: 4, 3: 6}
