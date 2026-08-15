from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from flight_delay.config import Paths
from flight_delay.ingest.extract import (
    TRAILING_FIELD,
    HeaderMismatchError,
    extract_month,
    read_header,
    staging_csv,
    verify_headers,
)

HEADER = "Year,Month,Origin,ArrDel15,"
ROW = '2023,1,"BDL","0.00",'


def _write_zip(paths: Paths, year: int, month: int, *, header: str = HEADER) -> Path:
    paths.raw.mkdir(parents=True, exist_ok=True)
    target = paths.raw_zip(year, month)
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr(f"On_Time_{year}_{month}.csv", f"{header}\n{ROW}\n")
        archive.writestr("readme.html", "<html></html>")
    return target


def test_trailing_empty_field_is_named(tmp_path: Path) -> None:
    """The feed ends every line with a comma. Leaving that field anonymous is
    how a positional schema silently shifts by one column."""
    paths = Paths(root=tmp_path)
    zip_path = _write_zip(paths, 2023, 1)
    assert read_header(zip_path) == ("Year", "Month", "Origin", "ArrDel15", TRAILING_FIELD)


def test_readme_member_is_ignored(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path)
    _write_zip(paths, 2023, 1)
    assert extract_month(paths, 2023, 1).bytes_written > 0


def test_extract_writes_hive_layout_and_is_idempotent(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path)
    _write_zip(paths, 2023, 4)

    first = extract_month(paths, 2023, 4)
    assert first.skipped is False
    assert first.csv_path == staging_csv(paths, 2023, 4)
    assert "Year=2023" in first.csv_path.parts[-3]
    assert "Month=4" in first.csv_path.parts[-2]
    assert first.csv_path.read_text().startswith("Year,Month,Origin,ArrDel15,")

    second = extract_month(paths, 2023, 4)
    assert second.skipped is True
    assert second.bytes_written == first.bytes_written


def test_extract_leaves_no_part_file_behind(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path)
    _write_zip(paths, 2023, 5)
    result = extract_month(paths, 2023, 5)
    assert not list(result.csv_path.parent.glob("*.part"))


def test_verify_headers_accepts_a_consistent_feed(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path)
    _write_zip(paths, 2023, 1)
    _write_zip(paths, 2023, 2)
    assert verify_headers(paths)[0] == "Year"


def test_verify_headers_rejects_a_reordered_month(tmp_path: Path) -> None:
    """A mid-year column reorder would corrupt the table with no error
    anywhere downstream, so it is caught before a byte is written."""
    paths = Paths(root=tmp_path)
    _write_zip(paths, 2023, 1)
    _write_zip(paths, 2023, 2, header="Year,Month,Origin,ArrDel15,Surprise,")

    with pytest.raises(HeaderMismatchError, match="2023-02"):
        verify_headers(paths)


def test_verify_headers_fails_loudly_on_an_empty_lake(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path)
    paths.raw.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        verify_headers(paths)
