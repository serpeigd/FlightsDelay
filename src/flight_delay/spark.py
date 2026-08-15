"""Spark session tuned for one laptop, not for a cluster.

Defaults built for a datacentre are actively harmful here. The machine has
8 threads and 7.6 GB of RAM inside WSL, so the two settings that matter are
the shuffle partition count -- 200 by default, which turns every shuffle into
hundreds of tiny tasks and files -- and driver memory, since in local mode the
driver *is* the executor.

Delta is optional on purpose. Its JARs are resolved from Maven at session
start, which needs network and is the most likely thing to fail; the fallback
is plain partitioned Parquet, which still demonstrates partition pruning.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from flight_delay.config import Paths

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

#: 2x the core count: enough to keep every thread busy without shredding the
#: data into tasks whose scheduling costs more than their work.
DEFAULT_SHUFFLE_PARTITIONS = 16
DEFAULT_DRIVER_MEMORY = "4g"

#: What actually turns Delta on. Fetching the JARs is necessary but not
#: sufficient.
DELTA_CONFIG = {
    "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
    "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
}


class DeltaUnavailableError(RuntimeError):
    """Delta could not be enabled; the caller should fall back to Parquet."""


def _base_config(paths: Paths, shuffle_partitions: int, driver_memory: str) -> dict[str, str]:
    return {
        "spark.driver.memory": driver_memory,
        "spark.sql.shuffle.partitions": str(shuffle_partitions),
        # Let Spark coalesce small post-shuffle partitions and switch join
        # strategies once it has seen real statistics.
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        # Shuffle spill is the heaviest writer in this project; keep it on the
        # Linux filesystem, never on /mnt/c.
        "spark.local.dir": os.environ.get("SPARK_LOCAL_DIRS", str(paths.root / ".spark-tmp")),
        "spark.sql.session.timeZone": "UTC",
        "spark.ui.showConsoleProgress": "false",
        # By default Spark reads "x" as the string literal 'x', not as the
        # identifier x. The same SQL that works on DuckDB then computes
        # avg('ArrDel15') over a constant string. Here it raised
        # CAST_INVALID_INPUT, but a numeric-looking column name would have
        # silently returned a wrong answer.
        "spark.sql.ansi.doubleQuotedIdentifiers": "true",
    }


def build_session(
    *,
    paths: Paths | None = None,
    app_name: str = "flight-delay",
    delta: bool = True,
    shuffle_partitions: int = DEFAULT_SHUFFLE_PARTITIONS,
    driver_memory: str = DEFAULT_DRIVER_MEMORY,
) -> SparkSession:
    """Create a local Spark session.

    With ``delta=True`` the Delta extensions are configured; if the JARs
    cannot be resolved, :class:`DeltaUnavailableError` is raised so the caller
    can decide rather than discovering it halfway through a write.
    """
    from pyspark.sql import SparkSession

    resolved = paths or Paths.from_env()
    builder = SparkSession.builder.appName(app_name).master("local[*]")
    for key, value in _base_config(resolved, shuffle_partitions, driver_memory).items():
        builder = builder.config(key, value)

    if not delta:
        return builder.getOrCreate()

    try:
        from delta import configure_spark_with_delta_pip
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise DeltaUnavailableError("delta-spark is not installed") from exc

    # configure_spark_with_delta_pip only adds the Maven coordinates; wiring
    # the SQL extension and the catalog is the caller's job. Without these two
    # the JARs download, the session starts, and the first Delta write fails
    # with DELTA_CONFIGURE_SPARK_SESSION_WITH_EXTENSION_AND_CATALOG.
    for key, value in DELTA_CONFIG.items():
        builder = builder.config(key, value)

    try:
        return configure_spark_with_delta_pip(builder).getOrCreate()
    except Exception as exc:  # pragma: no cover - depends on network
        raise DeltaUnavailableError(
            "could not start Spark with Delta enabled (the JARs are fetched "
            "from Maven at session start). Fall back to Parquet."
        ) from exc


@contextmanager
def session(**kwargs: Any) -> Iterator[SparkSession]:
    """Session that always stops, so a failed run cannot hold the JVM open."""
    spark = build_session(**kwargs)
    try:
        yield spark
    finally:
        spark.stop()
