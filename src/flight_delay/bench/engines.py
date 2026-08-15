"""The same SQL, the same data, two engines.

13.9M rows do not need Spark. Saying so is easy; the useful version is to run
identical work on both engines across growing slices of the data and publish
where, if anywhere, the lines cross.

The workloads below are plain SQL strings executed verbatim on DuckDB and on
Spark SQL. Nothing is rewritten per engine, so any difference in the numbers is
the engine and not a hand-tuned query. Both read the same curated Parquet.

Two things are deliberately kept honest:

- Spark's JVM startup is reported separately rather than folded into a query
  time. It is a real cost for a one-shot job and a rounding error for a long
  session, and averaging it into per-query numbers hides which case is being
  discussed.
- Each measurement is repeated and the median is reported, because at these
  sizes the data fits in the page cache and a single cold run measures the
  disk, not the engine.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

#: Month boundaries as Year*100+Month, so a scale is "the first N months".
_MONTHS: Sequence[int] = tuple(
    year * 100 + month for year in (2023, 2024) for month in range(1, 13)
)


def months_filter(n_months: int) -> str:
    """SQL predicate selecting the first ``n_months`` of the feed."""
    if not 1 <= n_months <= len(_MONTHS):
        raise ValueError(f"n_months must be in 1..{len(_MONTHS)}, got {n_months}")
    return f'("Year" * 100 + "Month") <= {_MONTHS[n_months - 1]}'


#: Grouped aggregation. This is the daily delay-rate series the time-series
#: model consumes, so it is real work rather than a synthetic benchmark.
DAILY_RATE_BY_AIRPORT = """
SELECT "Origin",
       "FlightDate",
       count(*)                                   AS flights,
       avg("ArrDel15")                            AS delay_rate,
       avg("ArrDelay")                            AS mean_delay
FROM {table}
WHERE {months} AND "ArrDel15" IS NOT NULL
GROUP BY "Origin", "FlightDate"
"""

#: Window function over aircraft rotation: how late was the previous leg flown
#: by this same airframe? A wide transformation -- it shuffles on Tail_Number --
#: and the feature most likely to carry real signal for a pre-departure model.
#:
#: The ORDER BY carries three tiebreakers beyond date and scheduled time, and
#: they are not decoration. 2,408 groups in the feed share a Tail_Number,
#: FlightDate and CRSDepTime -- 4,819 rows, physically impossible for one
#: airframe and so a reporting artifact. With ties, `lag` has no defined
#: answer: DuckDB and Spark each pick a different row as "previous" and the
#: two disagreed on real output. Adding Flight_Number, Origin and Dest makes
#: the ordering total, with zero remaining ties.
INBOUND_LEG_DELAY = """
SELECT count(*)                       AS rows,
       avg(prev_arr_delay)            AS mean_prev_delay,
       count(prev_arr_delay)          AS with_previous
FROM (
    SELECT lag("ArrDelay") OVER (
               PARTITION BY "Tail_Number"
               ORDER BY "FlightDate", "CRSDepTime",
                        "Flight_Number_Reporting_Airline", "Origin", "Dest"
           ) AS prev_arr_delay
    FROM {table}
    WHERE {months} AND "Tail_Number" IS NOT NULL
) t
"""

WORKLOADS: dict[str, str] = {
    "daily_rate_by_airport": DAILY_RATE_BY_AIRPORT,
    "inbound_leg_delay": INBOUND_LEG_DELAY,
}


@dataclass(frozen=True, slots=True)
class Timing:
    engine: str
    workload: str
    n_months: int
    seconds: float
    rows_out: int
    checksum: float | None

    @property
    def label(self) -> str:
        return f"{self.engine}/{self.workload}@{self.n_months}m"


def _repeat(
    fn: Callable[[], tuple[int, float | None]], repeats: int
) -> tuple[float, int, float | None]:
    """Run ``fn`` ``repeats`` times, return (median seconds, rows, checksum)."""
    times: list[float] = []
    rows = 0
    checksum: float | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        rows, checksum = fn()
        times.append(time.perf_counter() - start)
    return median(times), rows, checksum


def run_duckdb(
    con: Any, table_expr: str, workload: str, n_months: int, *, repeats: int = 3
) -> Timing:
    sql = WORKLOADS[workload].format(table=table_expr, months=months_filter(n_months))

    def once() -> tuple[int, float | None]:
        rows = con.execute(sql).fetchall()
        return len(rows), _checksum(rows)

    seconds, rows_out, checksum = _repeat(once, repeats)
    return Timing("duckdb", workload, n_months, seconds, rows_out, checksum)


def run_spark(
    spark: Any, table_expr: str, workload: str, n_months: int, *, repeats: int = 3
) -> Timing:
    sql = WORKLOADS[workload].format(table=table_expr, months=months_filter(n_months))

    def once() -> tuple[int, float | None]:
        rows = [tuple(r) for r in spark.sql(sql).collect()]
        return len(rows), _checksum(rows)

    seconds, rows_out, checksum = _repeat(once, repeats)
    return Timing("spark", workload, n_months, seconds, rows_out, checksum)


def _checksum(rows: Sequence[Sequence[Any]]) -> float | None:
    """Order-independent digest of the numeric columns of a result set.

    Used only to assert the two engines produced the same answer. Rounded
    because float accumulation order differs between them and an exact match
    would fail for reasons that have nothing to do with correctness.
    """
    total = 0.0
    for row in rows:
        for value in row:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += float(value)
    return round(total, 3)
