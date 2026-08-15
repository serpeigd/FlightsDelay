"""DuckDB curation: staged CSV to a partitioned Parquet dataset.

Every column is read as ``VARCHAR`` and cast explicitly afterwards. That is
slower than letting the reader sniff types, and it is the point: a sniffed
schema silently turns an unparseable value into a null, and nobody ever counts
them. Here the cast failures are counted per column and reported, so data
quality is a number rather than a hope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from flight_delay.config import Paths
from flight_delay.ingest.extract import TRAILING_FIELD, verify_headers
from flight_delay.ingest.schema import BY_NAME, KEPT_COLUMNS, PARTITION_BY, SqlType

if TYPE_CHECKING:
    import duckdb

#: The declared types of the contract, expressed in DuckDB's dialect.
_DUCKDB_TYPES: dict[SqlType, str] = {
    "string": "VARCHAR",
    "int": "INTEGER",
    "double": "DOUBLE",
    "date": "DATE",
}


def quote(identifier: str) -> str:
    """Quote an identifier. ``Year`` and ``Month`` are reserved-ish words and
    the feed's names are case-sensitive, so nothing goes unquoted."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


@dataclass(frozen=True, slots=True)
class CastFailure:
    column: str
    declared_type: str
    non_null_inputs: int
    failed: int

    @property
    def rate(self) -> float:
        return self.failed / self.non_null_inputs if self.non_null_inputs else 0.0


@dataclass(frozen=True, slots=True)
class CurationReport:
    rows_read: int
    rows_written: int
    cast_failures: tuple[CastFailure, ...]
    output: Path

    @property
    def clean(self) -> bool:
        return not self.cast_failures


def raw_columns_clause(header: tuple[str, ...]) -> str:
    """``columns=`` argument pinning all fields to VARCHAR, in feed order.

    Passing the column list explicitly, rather than letting the reader infer
    names from the header, is what keeps the phantom trailing field from
    shifting every column by one position.
    """
    entries = ", ".join(f"{quote(name)}: 'VARCHAR'" for name in header)
    return "{" + entries + "}"


def select_clause() -> str:
    """Explicit cast of each contract column, with the source name preserved."""
    parts = []
    for name in KEPT_COLUMNS:
        target = _DUCKDB_TYPES[BY_NAME[name].sql_type]
        # TRY_CAST yields NULL instead of aborting the whole load, which lets
        # the failures be counted rather than crashing on row 4 million.
        parts.append(f"TRY_CAST({quote(name)} AS {target}) AS {quote(name)}")
    return ",\n       ".join(parts)


def read_csv_expression(csv_glob: str, header: tuple[str, ...]) -> str:
    return (
        f"read_csv('{csv_glob}', "
        f"columns={raw_columns_clause(header)}, "
        "header=true, hive_partitioning=false)"
    )


def cast_failure_query(source: str) -> str:
    """One query counting, for every contract column, how many non-empty
    values fail their declared cast.

    Deliberately a single statement. The obvious implementation -- one query
    per column -- costs one full scan per column, which over 50 columns and
    5.9 GB of CSV means reading roughly 295 GB to answer a data-quality
    question. Two aggregates per column in one pass reads it once.
    """
    aggregates = ["count(*) AS row_count"]
    for name in KEPT_COLUMNS:
        target = _DUCKDB_TYPES[BY_NAME[name].sql_type]
        col = quote(name)
        present = f"{col} IS NOT NULL AND {col} <> ''"
        aggregates.append(f"count(*) FILTER (WHERE {present}) AS {quote(name + '__present')}")
        aggregates.append(
            f"count(*) FILTER (WHERE {present} AND TRY_CAST({col} AS {target}) IS NULL) "
            f"AS {quote(name + '__failed')}"
        )
    return "SELECT " + ", ".join(aggregates) + f" FROM {source}"


def profile(con: duckdb.DuckDBPyConnection, source: str) -> tuple[int, tuple[CastFailure, ...]]:
    """Row count and per-column cast failures, in a single scan."""
    row = con.execute(cast_failure_query(source)).fetchone()
    if row is None:  # pragma: no cover - count(*) always returns a row
        raise RuntimeError("cast-failure query returned no row")

    rows = int(row[0])
    failures: list[CastFailure] = []
    for index, name in enumerate(KEPT_COLUMNS):
        present, failed = int(row[1 + 2 * index]), int(row[2 + 2 * index])
        if failed:
            failures.append(
                CastFailure(name, _DUCKDB_TYPES[BY_NAME[name].sql_type], present, failed)
            )
    return rows, tuple(failures)


def curate(
    con: duckdb.DuckDBPyConnection,
    paths: Paths,
    *,
    table_name: str = "flights_duckdb",
    check_casts: bool = True,
) -> CurationReport:
    """Build the curated Parquet dataset from the CSV staging area."""
    header = verify_headers(paths)
    if header[-1] != TRAILING_FIELD:
        raise RuntimeError(f"unexpected final field {header[-1]!r}")

    csv_glob = str(paths.root / "staging" / "csv" / "*" / "*" / "part.csv")
    source = read_csv_expression(csv_glob, header)

    con.execute(f"CREATE OR REPLACE VIEW raw AS SELECT * FROM {source}")

    # Row count and cast profile share one scan of the CSV; the COPY below is
    # the only other pass over it.
    if check_casts:
        rows_read, failures = profile(con, "raw")
    else:
        rows_read = int(con.execute("SELECT count(*) FROM raw").fetchone()[0])  # type: ignore[index]
        failures = ()

    output = paths.table(table_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    partition_cols = ", ".join(quote(c) for c in PARTITION_BY)
    con.execute(
        f"COPY (SELECT {select_clause()} FROM raw) "
        f"TO '{output}' "
        f"(FORMAT PARQUET, PARTITION_BY ({partition_cols}), OVERWRITE_OR_IGNORE, "
        "COMPRESSION ZSTD)"
    )

    written = int(
        con.execute(f"SELECT count(*) FROM read_parquet('{output}/**/*.parquet')").fetchone()[0]  # type: ignore[index]
    )
    return CurationReport(rows_read, written, failures, output)
