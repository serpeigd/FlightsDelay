"""Shared plumbing for the pipeline commands."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from flight_delay.config import Paths, mlflow_artifact_uri, mlflow_tracking_uri

if TYPE_CHECKING:
    import duckdb

#: Leaves headroom under the 7.6 GB WSL gives us, so a spill goes to disk
#: rather than to the OOM killer.
MEMORY_LIMIT = "5GB"
THREADS = 8


def log(message: str = "") -> None:
    """Print immediately.

    Python block-buffers stdout when it is redirected, so a long-running step
    watched through a log file shows nothing at all until it finishes.
    """
    print(message, flush=True)


@contextmanager
def duckdb_connection(paths: Paths) -> Iterator[duckdb.DuckDBPyConnection]:
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    # Spill inside the WSL filesystem, never through /mnt/c.
    con.execute(f"SET temp_directory='{paths.root}/.duckdb-tmp'")
    try:
        yield con
    finally:
        con.close()


def start_experiment(paths: Paths, name: str) -> None:
    """Point MLflow at the lake and make sure the experiment exists.

    A SQLite backend does not imply an artifact location, and the default would
    be ``./mlruns`` next to the source tree -- inside OneDrive.
    """
    import mlflow

    paths.mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri(paths))
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=mlflow_artifact_uri(paths))
    mlflow.set_experiment(name)


def write_result(paths: Paths, name: str, payload: Any) -> None:
    paths.bench.mkdir(parents=True, exist_ok=True)
    target = paths.bench / name
    target.write_text(json.dumps(payload, indent=2))
    log(f"\nwrote {target}")


def require_table(paths: Paths, name: str, produced_by: str) -> str:
    """Glob for a curated table, with a message that says how to build it."""
    table = paths.table(name)
    if not table.is_dir():
        raise SystemExit(f"{table} does not exist. Run `flight-delay {produced_by}` first.")
    return f"read_parquet('{table}/**/*.parquet', hive_partitioning=true)"
