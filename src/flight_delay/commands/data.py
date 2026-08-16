"""Raw ZIPs to the curated table, and the model table built from it.

These three steps produce every figure in ``docs/data-profile.md``.
"""

from __future__ import annotations

import time

from flight_delay.commands._common import duckdb_connection, log
from flight_delay.config import Paths
from flight_delay.engines.duck import curate
from flight_delay.features.build import build
from flight_delay.ingest.extract import extract_all


def cmd_extract(paths: Paths, *, force: bool = False) -> int:
    """Unpack the monthly archives into the CSV staging area."""
    start = time.perf_counter()
    results = extract_all(paths, force=force)
    if not results:
        log(f"no archives under {paths.raw}; run scripts/download_bts.sh")
        return 1

    written = sum(r.bytes_written for r in results)
    skipped = sum(1 for r in results if r.skipped)
    for r in results:
        log(
            f"  {r.year}-{r.month:02d}  {r.bytes_written / 1e6:8.1f} MB  "
            f"{'already there' if r.skipped else 'extracted'}"
        )
    log(
        f"\n{len(results)} months, {written / 1e9:.2f} GB "
        f"({skipped} already present) in {time.perf_counter() - start:.0f}s"
    )
    return 0


def cmd_curate(paths: Paths) -> int:
    """Staged CSV to Parquet partitioned by year and month."""
    with duckdb_connection(paths) as con:
        start = time.perf_counter()
        report = curate(con, paths)
        elapsed = time.perf_counter() - start

    log(f"rows read     {report.rows_read:,}")
    log(f"rows written  {report.rows_written:,}")
    log(f"elapsed       {elapsed:.1f}s")
    log(f"output        {report.output}")

    if report.rows_read != report.rows_written:
        log("\nROW COUNT MISMATCH between input and output")
        return 1

    if report.cast_failures:
        log("\nCAST FAILURES (non-empty input that became NULL):")
        for f in report.cast_failures:
            log(
                f"  {f.column:32s} {f.declared_type:8s} "
                f"{f.failed:>10,} / {f.non_null_inputs:>12,}  ({f.rate:.4%})"
            )
        return 1

    log("\nno cast failures: every non-empty value parsed into its declared type")
    return 0


def cmd_features(paths: Paths) -> int:
    """Curated table to the modelling table, with the cutoff applied."""
    with duckdb_connection(paths) as con:
        start = time.perf_counter()
        report = build(con, paths)
        log(f"built in {time.perf_counter() - start:.1f}s")
        log(f"rows          {report.rows:,}")
        log(f"train (2023)  {report.train_rows:,}")
        log(f"test  (2024)  {report.test_rows:,}")
        log(f"inbound known {report.inbound_known_rate:.1%}")

        src = f"read_parquet('{report.output}/**/*.parquet', hive_partitioning=true)"
        log("\ndelay rate by inbound status:")
        rows = con.execute(
            f"""
            SELECT CASE WHEN inbound_delay IS NULL THEN 'not known at cutoff'
                        WHEN inbound_delay <= 0    THEN 'landed on time or early'
                        WHEN inbound_delay <= 15   THEN 'landed 1-15 min late'
                        WHEN inbound_delay <= 60   THEN 'landed 16-60 min late'
                        ELSE 'landed 60+ min late' END AS bucket,
                   count(*) AS flights,
                   round(100.0 * avg(label), 2) AS pct_delayed
            FROM {src} GROUP BY 1 ORDER BY pct_delayed
            """
        ).fetchall()
    for bucket, flights, pct in rows:
        log(f"  {bucket:26s} {flights:>11,}  {pct:5.2f}%")
    return 0
