"""Layout and engine benchmarks.

Produces every figure in ``docs/benchmarks.md``. These are the only commands
that need Java and a Spark session, so they are the ones to skip when the goal
is just to reproduce the modelling results.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from flight_delay.bench.engines import WORKLOADS, run_duckdb, run_spark
from flight_delay.bench.layout import describe, write_delta
from flight_delay.bench.scale import (
    FEED_START_YEAR,
    Projection,
    memory_verdict,
    probe_feed,
    rows_per_scale,
)
from flight_delay.bench.scan import ScanResult, measure
from flight_delay.commands._common import duckdb_connection, log, write_result
from flight_delay.config import YEARS, Paths
from flight_delay.spark import DeltaUnavailableError, session

#: Aggregating a column forces a real read. `count(*)` alone is answered from
#: per-file row counts in the Delta log without opening a single file.
AGG = ["count(*) AS n", "avg(ArrDelay) AS mean_delay"]

#: Enough rows per file that a partition holds several. With one file per
#: partition there is nothing for data skipping to skip.
MAX_RECORDS_PER_FILE = 50_000

SCALES = (1, 3, 6, 12, 24)
REPEATS = 3


def _show(r: ScanResult, baseline: ScanResult | None = None) -> None:
    line = f"  {r.label:32s} files={r.files_read:4d}  {r.megabytes_read:8.1f} MB  {r.seconds:6.2f}s"
    if baseline is not None:
        line += f"  skipped={r.skipping_vs(baseline):6.1%}"
    log(line)


def _as_dict(r: ScanResult) -> dict[str, Any]:
    return {
        "files": r.files_read,
        "megabytes": round(r.megabytes_read, 1),
        "seconds": round(r.seconds, 2),
    }


def cmd_bench_layout(paths: Paths) -> int:
    """Partition pruning, and clustering measured with one variable changed."""
    parquet = str(paths.table("flights_duckdb"))
    results: dict[str, Any] = {}

    try:
        spark_session = session(paths=paths, app_name="bench-layout")
    except DeltaUnavailableError as exc:
        log(f"Delta unavailable: {exc}")
        return 1

    with spark_session as spark:
        log("== building two tables that differ only in row order ==")
        tables = {}
        for name, cluster in (("flights_natural", ()), ("flights_clustered", ("origin",))):
            target = str(paths.table(name))
            start = time.perf_counter()
            write_delta(
                spark,
                parquet,
                target,
                max_records_per_file=MAX_RECORDS_PER_FILE,
                cluster_by=cluster,
            )
            detail = describe(spark, target)
            tables[name] = target
            log(
                f"  {name:20s} {detail.num_files:4d} files, {detail.megabytes:7.1f} MB "
                f"({time.perf_counter() - start:.0f}s)"
            )
            results[name] = {"files": detail.num_files, "megabytes": round(detail.megabytes, 1)}

        def query(target: str, predicate: str | None) -> Any:
            df = spark.read.format("delta").load(target)
            if predicate:
                df = df.filter(predicate)
            return df.selectExpr(*AGG)

        log("\n== partition pruning (Year/Month are partition columns) ==")
        full = measure("full table scan", query(tables["flights_natural"], None))
        _show(full)
        results["pruning"] = {"full": _as_dict(full)}
        for predicate in ("Year = 2024", "Year = 2024 AND Month = 7"):
            r = measure(predicate, query(tables["flights_natural"], predicate))
            _show(r, full)
            results["pruning"][predicate] = _as_dict(r)

        log("\n== clustering: same files, same bytes, different row order ==")
        results["clustering"] = {}
        for predicate in ("Origin = 'ATL'", "Origin = 'JFK'", "Origin IN ('ATL','SEA')"):
            natural = measure(predicate, query(tables["flights_natural"], predicate))
            clustered = measure(predicate, query(tables["flights_clustered"], predicate))
            skipped = (1 - clustered.bytes_read / natural.bytes_read) if natural.bytes_read else 0.0
            log(
                f"  {predicate:32s} {natural.files_read:4d}f {natural.megabytes_read:7.1f}MB  ->  "
                f"{clustered.files_read:4d}f {clustered.megabytes_read:7.1f}MB   "
                f"skipped={skipped:6.1%}   same answer: {natural.result == clustered.result}"
            )
            results["clustering"][predicate] = {
                "natural": _as_dict(natural),
                "clustered": _as_dict(clustered),
                "bytes_skipped_fraction": round(skipped, 4),
                "answers_match": natural.result == clustered.result,
            }

    write_result(paths, "layout.json", results)
    return 0


def cmd_bench_engines(paths: Paths) -> int:
    """The same SQL on DuckDB and Spark, at growing slices of the feed."""
    parquet = str(paths.table("flights_duckdb"))
    results: dict[str, Any] = {"scales_months": list(SCALES), "repeats": REPEATS}
    timings: dict[tuple[str, str, int], Any] = {}

    log("== DuckDB ==")
    start = time.perf_counter()
    with duckdb_connection(paths) as con:
        results["duckdb_startup_seconds"] = round(time.perf_counter() - start, 3)
        table = f"read_parquet('{parquet}/**/*.parquet', hive_partitioning=true)"
        for workload in WORKLOADS:
            for months in SCALES:
                t = run_duckdb(con, table, workload, months, repeats=REPEATS)
                timings[("duckdb", workload, months)] = t
                log(f"  {t.label:44s} {t.seconds:7.2f}s  rows_out={t.rows_out:>8,}")

    log("\n== Spark ==")
    start = time.perf_counter()
    with session(paths=paths, app_name="bench-engines", delta=False) as spark:
        results["spark_startup_seconds"] = round(time.perf_counter() - start, 2)
        log(f"  session startup: {results['spark_startup_seconds']}s")
        spark.read.parquet(parquet).createOrReplaceTempView("flights")
        for workload in WORKLOADS:
            for months in SCALES:
                t = run_spark(spark, "flights", workload, months, repeats=REPEATS)
                timings[("spark", workload, months)] = t
                log(f"  {t.label:44s} {t.seconds:7.2f}s  rows_out={t.rows_out:>8,}")

    log("\n== side by side (median of 3, seconds) ==")
    results["timings"] = {}
    mismatches = 0
    for workload in WORKLOADS:
        log(f"\n  {workload}")
        log(f"    {'months':>6}  {'duckdb':>9}  {'spark':>9}  {'ratio':>7}   answers match")
        for months in SCALES:
            d = timings[("duckdb", workload, months)]
            s = timings[("spark", workload, months)]
            ratio = s.seconds / d.seconds if d.seconds else float("nan")
            match = d.rows_out == s.rows_out and d.checksum == s.checksum
            mismatches += 0 if match else 1
            log(
                f"    {months:>6}  {d.seconds:8.2f}s  {s.seconds:8.2f}s  {ratio:6.1f}x   "
                f"{match}  ({d.rows_out:,} rows)"
            )
            results["timings"][f"{workload}@{months}m"] = {
                "duckdb_seconds": round(d.seconds, 3),
                "spark_seconds": round(s.seconds, 3),
                "spark_over_duckdb": round(ratio, 2),
                "rows_out": d.rows_out,
                "answers_match": match,
            }

    write_result(paths, "engines.json", results)
    if mismatches:
        log(f"\n{mismatches} scale(s) where the two engines disagreed")
        return 1
    return 0


def _directory_bytes(directory: Path, pattern: str) -> int:
    return sum(f.stat().st_size for f in directory.rglob(pattern) if f.is_file())


def _machine_bytes() -> int:
    """Physical memory, or 0 where the platform will not say."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def cmd_scale(paths: Paths) -> int:
    """How large is the feed really, and where does one machine run out?

    The engine benchmark measures five points and refuses to extrapolate a
    crossing point. This command measures the axis those points sit on: the
    published size of every month in the feed, taken from the server rather
    than assumed, and what the whole of it would weigh curated.
    """
    parquet = paths.table("flights_duckdb")
    if not parquet.is_dir():
        log("no curated table; run `flight-delay curate` first")
        return 1

    table = f"read_parquet('{parquet}/**/*.parquet', hive_partitioning=true)"
    with duckdb_connection(paths) as con:
        per_scale = rows_per_scale(con, table, SCALES)
        counted = con.execute(f"SELECT count(*) FROM {table}").fetchone()
        rows_total = int(counted[0]) if counted else 0

    log("== rows actually scanned at each benchmark scale ==")
    for months, rows in per_scale.items():
        log(f"  {months:>2} months  {rows:>12,} rows")

    parquet_bytes = _directory_bytes(parquet, "*.parquet")
    zip_bytes = _directory_bytes(paths.raw, "*.zip")
    log(f"\ncurated parquet   {parquet_bytes / 1e6:8.1f} MB for {rows_total:,} rows")
    log(f"downloaded ZIPs   {zip_bytes / 1e6:8.1f} MB")

    years = range(FEED_START_YEAR, max(YEARS) + 1)
    log(f"\n== probing every monthly archive, {FEED_START_YEAR}-{max(YEARS)} (HEAD only) ==")
    start = time.perf_counter()
    feed = probe_feed(years)
    log(f"  {feed.found} of {feed.probed} months answered in {time.perf_counter() - start:.0f}s")
    if feed.found == 0:
        log("  nothing answered — the projection needs the network, so it is skipped")
        write_result(paths, "scale.json", {"rows_per_scale": per_scale, "feed": None})
        return 1
    log(f"  span              {feed.first_month} to {feed.last_month}")
    log(f"  published total   {feed.compressed_gigabytes:.2f} GB compressed")
    if feed.absent_years:
        log(f"  no archive under this name for: {', '.join(str(y) for y in feed.absent_years)}")

    projection = Projection(
        rows_measured=rows_total,
        compressed_bytes_measured=zip_bytes,
        parquet_bytes_measured=parquet_bytes,
        feed_compressed_bytes=feed.compressed_bytes,
    )
    log("\n== projecting the whole feed from what was measured on 24 months ==")
    log(f"  {projection.rows_per_compressed_byte * 1e9:,.0f} rows per compressed GB")
    log(f"  {projection.parquet_bytes_per_row:.2f} bytes of curated Parquet per row")
    log(f"  => {projection.estimated_feed_rows:,} rows (estimate)")
    log(f"  => {projection.estimated_feed_parquet_bytes / 1e9:.2f} GB curated (estimate)")
    log(f"  => {projection.multiple_of_measured:.1f}x what this project used")

    machine = _machine_bytes()
    verdict = memory_verdict(projection, machine_bytes=machine)
    log("\n== does the whole feed still fit on one machine? ==")
    log(f"  this machine       {machine / 1e9:.1f} GB")
    log(
        f"  working set        {verdict['estimated_working_set_bytes'] / 1e9:.1f} GB "
        f"(Parquet x{verdict['expansion_assumed']:.0f}, assumed)"
    )
    log(f"  fits               {verdict['fits']}")
    log(f"  runs out at        ~{verdict['rows_that_fit']:,} rows")

    write_result(
        paths,
        "scale.json",
        {
            "rows_per_scale": {str(k): v for k, v in per_scale.items()},
            "feed": {
                "months_probed": feed.probed,
                "months_found": feed.found,
                "first_month": list(feed.first_month) if feed.first_month else None,
                "last_month": list(feed.last_month) if feed.last_month else None,
                "compressed_bytes": feed.compressed_bytes,
                "absent_years": list(feed.absent_years),
            },
            "projection": projection.as_dict(),
            "memory": verdict,
        },
    )
    return 0
