"""Build the Delta table and change its physical layout.

Partitioning and Z-ordering solve different problems, and the benchmark exists
to show that rather than assert it.

Partitioning by ``Year``/``Month`` puts each month in its own directory, so a
filter on those columns skips whole directories without opening a file. It only
helps for the columns chosen up front, and choosing too many produces the small
files problem instead.

``Origin`` cannot be a partition column -- around 380 airports times 24 months
would shatter the table -- so a filter on it reads everything. Z-ordering sorts
rows inside each partition so that similar ``Origin`` values land in the same
file, which narrows the per-file min/max statistics Delta keeps in its
transaction log, and lets the reader skip files whose range cannot match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flight_delay.ingest.schema import PARTITION_BY


@dataclass(frozen=True, slots=True)
class TableDetail:
    num_files: int
    size_bytes: int
    partition_columns: tuple[str, ...]

    @property
    def megabytes(self) -> float:
        return self.size_bytes / 1e6


def write_delta(
    spark: Any,
    source_parquet: str,
    target: str,
    *,
    partition_by: tuple[str, ...] = PARTITION_BY,
    max_records_per_file: int | None = None,
    cluster_by: tuple[str, ...] = (),
) -> None:
    """Load the curated Parquet and rewrite it as a partitioned Delta table.

    ``max_records_per_file`` caps file size. It matters for the clustering
    benchmark: data skipping decides per *file*, so a partition holding a
    single file has nothing to skip no matter how its rows are sorted.

    ``cluster_by`` sorts rows within each partition so that similar values land
    in the same file. That is the mechanism Z-ordering uses -- narrow per-file
    min/max statistics let a reader skip files whose range cannot match -- and
    sorting directly makes it measurable while holding file count and file size
    fixed, which ``OPTIMIZE ZORDER`` does not.
    """
    source = spark.read.parquet(source_parquet)
    if cluster_by:
        source = source.sortWithinPartitions(*cluster_by)

    writer = (
        source.write.format("delta")
        .partitionBy(*partition_by)
        .mode("overwrite")
        .option("overwriteSchema", "true")
    )
    if max_records_per_file is not None:
        writer = writer.option("maxRecordsPerFile", max_records_per_file)
    writer.save(target)


def describe(spark: Any, target: str) -> TableDetail:
    row = spark.sql(f"DESCRIBE DETAIL delta.`{target}`").collect()[0]
    return TableDetail(
        num_files=int(row["numFiles"]),
        size_bytes=int(row["sizeInBytes"]),
        partition_columns=tuple(row["partitionColumns"]),
    )


def set_target_file_size(spark: Any, target: str, size: str) -> None:
    """Set ``delta.targetFileSize`` on the table, e.g. ``'4mb'``.

    Kept for the record, and **not usable on open-source Delta**: 4.3.1 rejects
    the property with ``DELTA_UNKNOWN_CONFIGURATION`` because it is a
    Databricks-only setting. The session config
    ``spark.databricks.delta.optimize.maxFileSize`` is not an equivalent
    either — with it set, ``OPTIMIZE`` still compacted 291 files into one per
    partition. Neither knob makes ``OPTIMIZE ZORDER`` measurable at this table
    size; see ``docs/benchmarks.md`` for the experiment that replaced it.
    """
    spark.sql(f"ALTER TABLE delta.`{target}` SET TBLPROPERTIES ('delta.targetFileSize' = '{size}')")


def optimize_zorder(spark: Any, target: str, columns: tuple[str, ...]) -> dict[str, Any]:
    """Compact and Z-order the table, returning OPTIMIZE's own metrics."""
    by = ", ".join(columns)
    row = spark.sql(f"OPTIMIZE delta.`{target}` ZORDER BY ({by})").collect()[0]
    metrics = row["metrics"]
    return {
        "files_added": metrics["numFilesAdded"],
        "files_removed": metrics["numFilesRemoved"],
        "partitions_optimized": metrics["partitionsOptimized"],
    }
