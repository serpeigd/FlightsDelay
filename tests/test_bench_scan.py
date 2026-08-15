from __future__ import annotations

import pytest

from flight_delay.bench.scan import ScanResult


def result(**kwargs: object) -> ScanResult:
    defaults: dict[str, object] = {
        "label": "q",
        "files_read": 10,
        "bytes_read": 1_000_000,
        "rows_scanned": 500,
        "partitions_read": 2,
        "seconds": 1.0,
    }
    defaults.update(kwargs)
    return ScanResult(**defaults)  # type: ignore[arg-type]


def test_megabytes_uses_decimal_units() -> None:
    assert result(bytes_read=449_500_000).megabytes_read == pytest.approx(449.5)


def test_skipping_is_measured_in_bytes_not_files() -> None:
    """A query can touch every file and still read far fewer bytes, so the
    skipping ratio is deliberately byte-based."""
    baseline = result(files_read=291, bytes_read=444_700_000)
    clustered = result(files_read=50, bytes_read=79_500_000)
    assert clustered.skipping_vs(baseline) == pytest.approx(0.821, abs=0.001)


def test_no_skipping_when_the_same_bytes_are_read() -> None:
    baseline = result(bytes_read=444_700_000)
    assert result(bytes_read=444_700_000).skipping_vs(baseline) == pytest.approx(0.0)


def test_skipping_against_an_empty_baseline_is_zero_not_a_crash() -> None:
    assert result(bytes_read=0).skipping_vs(result(bytes_read=0)) == 0.0


def test_timing_is_excluded_from_equality() -> None:
    """Two runs of the same query differ in wall-clock but must compare equal
    on what they actually read."""
    assert result(seconds=2.10, result=("x",)) != result(seconds=0.48, result=("x",))
    assert result(result=("x",)) == result(result=("y",))
