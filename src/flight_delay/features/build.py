"""Build the modelling table with point-in-time correctness enforced in SQL.

Two features here would leak if written the obvious way, and both are written
the careful way instead.

**The inbound leg.** Propagated delay is the single largest cause of late
arrivals, so "how late did this airframe's previous flight land?" is the most
valuable pre-departure feature available. It is also a trap: the previous leg
may still be in the air at the moment the prediction is made. The value is
therefore admitted only when the previous leg's *actual arrival timestamp* is
at or before the cutoff, and a companion flag records whether it was known --
because "the inbound is not in yet" is itself a signal, and dropping those rows
would quietly delete the hardest cases.

Building that arrival timestamp needs care of its own. ``FlightDate`` is the
departure date and 4.76% of flights (654,296) land on the following calendar
day, so an arrival clock earlier than the departure clock means midnight was
crossed. ``'2400'`` is also a real value, appearing once in ``CRSDepTime`` and
7,705 times in ``ArrTime``.

**Historical congestion.** An airport's delay rate is a strong prior, but
computing it over the whole dataset leaks the future into the past. It is taken
from the *previous calendar month* only, which is genuinely knowable at
prediction time and leaves 2023-01 without a prior rather than pretending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from flight_delay.config import Paths

if TYPE_CHECKING:
    import duckdb

#: How far ahead of scheduled departure the prediction is made. Everything not
#: knowable at this moment is excluded from scenario A.
CUTOFF_HOURS: Final = 2

#: Beyond this gap the previous leg is not an inbound aircraft, it is an
#: airframe coming back from storage: the longest gap in the feed is 664 days.
#: Measured correlation between the previous leg's arrival delay and the label
#: peaks at a 12-24h gap (0.22) and collapses past three days (0.013), so the
#: cut is at a day. 309,894 rows lose the feature and keep the flag.
MAX_TURNAROUND_HOURS: Final = 24

#: Features knowable from the published schedule alone.
SCENARIO_A_FEATURES: Final[tuple[str, ...]] = (
    "dep_minute_of_day",
    "dep_hour",
    "arr_minute_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "distance",
    "crs_elapsed",
    "carrier",
    "origin",
    "dest",
    "origin_prior_rate",
    "origin_prior_flights",
    "carrier_prior_rate",
    "inbound_delay",
    "inbound_known",
    "inbound_turnaround_minutes",
)

#: Scenario B additionally knows the aircraft actually left, and how late.
SCENARIO_B_EXTRA: Final[tuple[str, ...]] = ("dep_delay", "taxi_out")
SCENARIO_B_FEATURES: Final[tuple[str, ...]] = SCENARIO_A_FEATURES + SCENARIO_B_EXTRA

CATEGORICAL: Final[tuple[str, ...]] = ("carrier", "origin", "dest")

LABEL: Final = "label"


@dataclass(frozen=True, slots=True)
class BuildReport:
    rows: int
    train_rows: int
    test_rows: int
    inbound_known_rate: float
    output: str


def _clock_to_minutes(column: str) -> str:
    """Zero-padded clock string to minutes past midnight.

    '2400' is a real value meaning end of day; 24*60 is left as-is rather than
    wrapped to 0, so ordering stays monotonic within a day.
    """
    return f"CAST(substr({column}, 1, 2) AS INTEGER) * 60 + CAST(substr({column}, 3, 2) AS INTEGER)"


def _timestamp(date_column: str, clock_column: str) -> str:
    return f"CAST({date_column} AS TIMESTAMP) + INTERVAL ({_clock_to_minutes(clock_column)}) MINUTE"


def model_table_sql(source: str) -> str:
    """SQL producing one row per flight with the cutoff already applied."""
    sched_dep = _timestamp('"FlightDate"', '"CRSDepTime"')
    arr_clock = _timestamp('"FlightDate"', '"ArrTime"')
    # An arrival clock earlier than the departure clock means the flight
    # crossed midnight, and FlightDate is the departure date. 4.76% of flights.
    actual_arr = (
        'CASE WHEN "ArrTime" IS NULL OR "DepTime" IS NULL THEN NULL '
        f'ELSE {arr_clock} + (CASE WHEN "ArrTime" < "DepTime" '
        "THEN INTERVAL 1 DAY ELSE INTERVAL 0 DAY END) END"
    )

    return f"""
WITH parsed AS (
    SELECT
        "Year", "Month", "FlightDate", "DayOfWeek",
        "Reporting_Airline"              AS carrier,
        "Origin"                         AS origin,
        "Dest"                           AS dest,
        "Tail_Number",
        "Flight_Number_Reporting_Airline",
        "Distance"                       AS distance,
        "CRSElapsedTime"                 AS crs_elapsed,
        "ArrDel15"                       AS {LABEL},
        "ArrDelay",
        "DepDelay"                       AS dep_delay,
        "TaxiOut"                        AS taxi_out,
        "Cancelled", "Diverted",
        {_clock_to_minutes('"CRSDepTime"')} AS dep_minute_of_day,
        {_clock_to_minutes('"CRSArrTime"')} AS arr_minute_of_day,
        {sched_dep}                      AS sched_dep_ts,
        {actual_arr}                     AS actual_arr_ts
    FROM {source}
),
-- Delay rate per origin and per carrier, by calendar month. Consumed only by
-- the *following* month, so nothing from the future reaches a feature.
origin_month AS (
    SELECT origin, "Year" * 12 + "Month" AS month_index,
           avg({LABEL}) AS rate, count(*) AS flights
    FROM parsed WHERE {LABEL} IS NOT NULL GROUP BY 1, 2
),
carrier_month AS (
    SELECT carrier, "Year" * 12 + "Month" AS month_index, avg({LABEL}) AS rate
    FROM parsed WHERE {LABEL} IS NOT NULL GROUP BY 1, 2
),
rotation AS (
    SELECT *,
        lag(actual_arr_ts) OVER w AS prev_arr_ts,
        lag("ArrDelay")    OVER w AS prev_arr_delay
    FROM parsed
    -- Total ordering: 4,819 rows tie on (tail, date, scheduled time), and with
    -- ties lag() is non-deterministic across engines.
    WINDOW w AS (
        PARTITION BY "Tail_Number"
        ORDER BY sched_dep_ts, "Flight_Number_Reporting_Airline", origin, dest
    )
),
usable AS (
    SELECT *,
        (prev_arr_ts IS NOT NULL
         AND prev_arr_ts <= sched_dep_ts - INTERVAL {CUTOFF_HOURS} HOUR
         AND prev_arr_ts >= sched_dep_ts - INTERVAL {MAX_TURNAROUND_HOURS} HOUR
        ) AS inbound_usable
    FROM rotation
)
SELECT
    r."Year"                                       AS year,
    r."FlightDate"                                 AS flight_date,
    r.dep_minute_of_day,
    r.dep_minute_of_day / 60                       AS dep_hour,
    r.arr_minute_of_day,
    r."DayOfWeek"                                  AS day_of_week,
    r."Month"                                      AS month,
    CAST(r."DayOfWeek" IN (6, 7) AS INTEGER)       AS is_weekend,
    r.distance,
    r.crs_elapsed,
    r.carrier,
    r.origin,
    r.dest,
    om.rate                                        AS origin_prior_rate,
    om.flights                                     AS origin_prior_flights,
    cm.rate                                        AS carrier_prior_rate,
    -- The inbound leg counts only if it had already landed by the cutoff and
    -- the gap is short enough to still mean "the aircraft that is coming".
    CASE WHEN r.inbound_usable THEN r.prev_arr_delay END      AS inbound_delay,
    CAST(r.inbound_usable AS INTEGER)                         AS inbound_known,
    CASE WHEN r.inbound_usable
         THEN date_diff('minute', r.prev_arr_ts, r.sched_dep_ts) END
                                                              AS inbound_turnaround_minutes,
    r.dep_delay,
    r.taxi_out,
    r.{LABEL}
FROM usable r
LEFT JOIN origin_month om
       ON om.origin = r.origin
      AND om.month_index = r."Year" * 12 + r."Month" - 1
LEFT JOIN carrier_month cm
       ON cm.carrier = r.carrier
      AND cm.month_index = r."Year" * 12 + r."Month" - 1
-- Cancelled and diverted flights carry no label; they are excluded here and
-- counted in docs/data-profile.md rather than dropped silently.
WHERE r.{LABEL} IS NOT NULL
"""


def build(
    con: duckdb.DuckDBPyConnection, paths: Paths, *, table_name: str = "model_table"
) -> BuildReport:
    source = f"read_parquet('{paths.table('flights_duckdb')}/**/*.parquet', hive_partitioning=true)"
    output = paths.table(table_name)
    output.parent.mkdir(parents=True, exist_ok=True)

    con.execute(f"CREATE OR REPLACE VIEW model_rows AS {model_table_sql(source)}")
    con.execute(
        f"COPY (SELECT * FROM model_rows) TO '{output}' "
        "(FORMAT PARQUET, PARTITION_BY (year), OVERWRITE_OR_IGNORE, COMPRESSION ZSTD)"
    )

    stats = con.execute(
        f"""
        SELECT count(*),
               count(*) FILTER (WHERE year = 2023),
               count(*) FILTER (WHERE year = 2024),
               avg(inbound_known)
        FROM read_parquet('{output}/**/*.parquet', hive_partitioning=true)
        """
    ).fetchone()
    assert stats is not None
    return BuildReport(
        rows=int(stats[0]),
        train_rows=int(stats[1]),
        test_rows=int(stats[2]),
        inbound_known_rate=float(stats[3]),
        output=str(output),
    )
