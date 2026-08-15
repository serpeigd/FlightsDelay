"""Unpack the monthly BTS ZIPs into a CSV staging area.

Spark cannot read ZIP archives, and ``unzip`` is not installed in this WSL
image (installing it needs sudo, which prompts for a password). Python's
``zipfile`` does the job and streams, so a 243 MB member never lands in memory
in one piece.

The header is verified against every archive rather than trusted from the
first one: a silent column reorder mid-year would otherwise corrupt the whole
table with no error anywhere.
"""

from __future__ import annotations

import csv
import io
import shutil
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from flight_delay.config import MONTHS, YEARS, Paths

#: Every header and data line ends with a comma, producing one unnamed
#: trailing field. It carries no data but must be accounted for, or a
#: positional schema silently shifts by one column.
TRAILING_FIELD = "_trailing"

_CHUNK = 8 << 20


class HeaderMismatchError(RuntimeError):
    """Raised when one month's columns differ from the rest."""


@dataclass(frozen=True, slots=True)
class ExtractedMonth:
    year: int
    month: int
    csv_path: Path
    bytes_written: int
    skipped: bool


def _member_name(archive: zipfile.ZipFile) -> str:
    members = [i.filename for i in archive.infolist() if i.filename.lower().endswith(".csv")]
    if len(members) != 1:
        raise RuntimeError(f"expected exactly one CSV member, found {members}")
    return members[0]


def read_header(zip_path: Path) -> tuple[str, ...]:
    """Column names of one archive, with the empty trailing field named."""
    with zipfile.ZipFile(zip_path) as archive, archive.open(_member_name(archive)) as raw:
        line = next(csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig")))
    if line and line[-1] == "":
        line = [*line[:-1], TRAILING_FIELD]
    return tuple(line)


def verify_headers(paths: Paths, expected: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Check that all present months share one header. Returns it."""
    reference: tuple[str, ...] | None = expected
    for year, month in available_months(paths):
        header = read_header(paths.raw_zip(year, month))
        if reference is None:
            reference = header
        elif header != reference:
            differing = set(header) ^ set(reference)
            raise HeaderMismatchError(
                f"{year}-{month:02d} header differs from the reference; "
                f"symmetric difference: {sorted(differing)}"
            )
    if reference is None:
        raise FileNotFoundError(f"no raw archives under {paths.raw}")
    return reference


def available_months(paths: Paths) -> Iterator[tuple[int, int]]:
    for year in YEARS:
        for month in MONTHS:
            if paths.raw_zip(year, month).is_file():
                yield year, month


def staging_csv(paths: Paths, year: int, month: int) -> Path:
    """Hive-style layout so Spark can read the whole tree in one go."""
    return paths.root / "staging" / "csv" / f"Year={year}" / f"Month={month}" / "part.csv"


def extract_month(paths: Paths, year: int, month: int, *, force: bool = False) -> ExtractedMonth:
    zip_path = paths.raw_zip(year, month)
    target = staging_csv(paths, year, month)

    if target.is_file() and not force:
        return ExtractedMonth(year, month, target, target.stat().st_size, skipped=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".csv.part")
    with (
        zipfile.ZipFile(zip_path) as archive,
        archive.open(_member_name(archive)) as src,
        partial.open("wb") as dst,
    ):
        shutil.copyfileobj(src, dst, _CHUNK)
    # Rename only after a complete write: a half-written file that looks
    # finished is how a truncated month slipped through the download step.
    partial.replace(target)
    return ExtractedMonth(year, month, target, target.stat().st_size, skipped=False)


def extract_all(paths: Paths, *, force: bool = False) -> list[ExtractedMonth]:
    verify_headers(paths)
    return [extract_month(paths, y, m, force=force) for y, m in available_months(paths)]
