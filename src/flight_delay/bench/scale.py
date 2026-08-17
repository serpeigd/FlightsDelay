"""How big is the feed really, and where would one machine stop being enough?

The engine benchmark answers "does 13.9M rows need Spark?" with a measurement.
It cannot answer "would it ever?", because two years is a slice of a feed that
starts in October 1987 — and an extrapolation from five converging points would
be a guess wearing a number's clothes.

This module measures the missing half instead of guessing it:

- **The size of the whole feed**, by issuing one HTTP ``HEAD`` per monthly
  archive and reading ``Content-Length``. No download, no assumption: the
  server states the size of every file the project did not take.
- **Rows per compressed byte**, calibrated on the 24 archives actually
  downloaded, where both numbers are known exactly.
- **Bytes per row of curated Parquet**, measured on disk.

Multiplying those gives an estimate of the full feed in rows and in curated
bytes, which is what the memory argument needs. The estimate is reported as an
estimate; the two inputs to it are measurements.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

#: The published archive is one ZIP per month under a stable name.
ARCHIVE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

#: The feed opens in October 1987. Probing from January of that year costs nine
#: requests and proves the start date rather than assuming it.
FEED_START_YEAR = 1987

HeadFn = Callable[[str], int]


def head_size(url: str, *, timeout: float = 30.0) -> int:
    """Content-Length of ``url``, or 0 if it is not published.

    A missing month is a 404 and a normal outcome — the PREZIP directory does
    not carry every month the agency has ever published. Anything else is also
    reported as absent rather than raised, because one flaky request should not
    lose the other 400 measurements; the count of what answered is published
    alongside the total so a partial sweep is visible.
    """
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.headers.get("Content-Length") or 0)
    except Exception:
        return 0


@dataclass(frozen=True, slots=True)
class FeedProbe:
    """What the publisher says it holds, without downloading any of it."""

    probed: int
    found: int
    compressed_bytes: int
    first_month: tuple[int, int] | None
    last_month: tuple[int, int] | None
    #: Years inside the probed span where no archive answered. Named rather
    #: than silently folded into the total, because a gap changes what the
    #: estimate below means.
    absent_years: tuple[int, ...]

    @property
    def compressed_gigabytes(self) -> float:
        return self.compressed_bytes / 1e9


def probe_feed(
    years: Iterable[int],
    *,
    workers: int = 6,
    head: HeadFn = head_size,
) -> FeedProbe:
    """One HEAD per month across ``years``. Concurrency is kept low on purpose."""
    months = [(year, month) for year in years for month in range(1, 13)]
    urls = [ARCHIVE_URL.format(year=y, month=m) for y, m in months]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        sizes = list(pool.map(head, urls))

    found = [(y, m, size) for (y, m), size in zip(months, sizes, strict=True) if size > 0]
    present_years = {y for y, _, _ in found}
    absent = tuple(y for y in sorted({y for y, _ in months}) if y not in present_years)
    return FeedProbe(
        probed=len(months),
        found=len(found),
        compressed_bytes=sum(size for _, _, size in found),
        first_month=found[0][:2] if found else None,
        last_month=found[-1][:2] if found else None,
        absent_years=absent,
    )


@dataclass(frozen=True, slots=True)
class Projection:
    """The feed expressed in the units the memory argument actually needs."""

    rows_measured: int
    compressed_bytes_measured: int
    parquet_bytes_measured: int
    feed_compressed_bytes: int

    @property
    def rows_per_compressed_byte(self) -> float:
        return self.rows_measured / self.compressed_bytes_measured

    @property
    def parquet_bytes_per_row(self) -> float:
        return self.parquet_bytes_measured / self.rows_measured

    @property
    def estimated_feed_rows(self) -> int:
        return round(self.feed_compressed_bytes * self.rows_per_compressed_byte)

    @property
    def estimated_feed_parquet_bytes(self) -> int:
        return round(self.estimated_feed_rows * self.parquet_bytes_per_row)

    @property
    def multiple_of_measured(self) -> float:
        return self.estimated_feed_rows / self.rows_measured

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_measured": self.rows_measured,
            "compressed_bytes_measured": self.compressed_bytes_measured,
            "parquet_bytes_measured": self.parquet_bytes_measured,
            "feed_compressed_bytes": self.feed_compressed_bytes,
            "rows_per_compressed_gigabyte": round(self.rows_per_compressed_byte * 1e9),
            "parquet_bytes_per_row": round(self.parquet_bytes_per_row, 2),
            "estimated_feed_rows": self.estimated_feed_rows,
            "estimated_feed_parquet_bytes": self.estimated_feed_parquet_bytes,
            "multiple_of_measured": round(self.multiple_of_measured, 2),
        }


def memory_verdict(
    projection: Projection, *, machine_bytes: int, expansion: float = 3.0
) -> dict[str, Any]:
    """Does the whole feed still fit in one machine?

    ``expansion`` is the factor between Parquet on disk and the working set of
    the window-function workload, which decodes the columns it touches and then
    shuffles them. It is an assumption, stated here rather than buried: the
    verdict is reported for the factor used, and the row count at which the
    machine runs out is reported with it so a different assumption can be
    applied by anyone who disagrees.
    """
    working_set = projection.estimated_feed_parquet_bytes * expansion
    per_row = projection.parquet_bytes_per_row * expansion
    return {
        "machine_bytes": machine_bytes,
        "expansion_assumed": expansion,
        "estimated_working_set_bytes": round(working_set),
        "fits": working_set < machine_bytes,
        "rows_that_fit": round(machine_bytes / per_row),
    }


def rows_per_scale(con: Any, table_expr: str, scales: Sequence[int]) -> dict[int, int]:
    """Exact input rows for each benchmark scale.

    The benchmark records rows *out*; the x axis of any chart that wants to
    generalise beyond this feed needs rows *in*. Months are not uniform — the
    first month is 538,837 rows, not a twenty-fourth of the total — so this is
    counted rather than divided.
    """
    from flight_delay.bench.engines import months_filter

    counts: dict[int, int] = {}
    for months in scales:
        sql = f"SELECT count(*) FROM {table_expr} WHERE {months_filter(months)}"
        counts[months] = int(con.execute(sql).fetchone()[0])
    return counts
