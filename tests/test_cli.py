from __future__ import annotations

from pathlib import Path

import pytest

from flight_delay.cli import main
from flight_delay.config import MONTHS, YEARS, Paths


def _make_complete_lake(root: Path) -> Paths:
    paths = Paths(root=root)
    paths.raw.mkdir(parents=True)
    for year in YEARS:
        for month in MONTHS:
            paths.raw_zip(year, month).write_bytes(b"x")
    return paths


def test_status_succeeds_on_a_complete_lake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_complete_lake(tmp_path)
    monkeypatch.setenv("FLIGHT_DELAY_DATA", str(tmp_path))
    assert main(["status"]) == 0


def test_status_flags_an_unfinished_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A truncated ``.part`` once passed the old `ls | wc -l` check because it
    still counted as a directory entry. Counting entries is not counting data."""
    paths = _make_complete_lake(tmp_path)
    paths.raw_zip(2024, 12).unlink()
    (paths.raw / "2024_12.zip.part").write_bytes(b"partial")

    monkeypatch.setenv("FLIGHT_DELAY_DATA", str(tmp_path))
    assert main(["status"]) == 1

    out = capsys.readouterr().out
    assert "INCOMPLETE" in out
    assert "2024_12.zip.part" in out
    assert "2024_12" in out  # also reported as missing


def test_status_reports_missing_months(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _make_complete_lake(tmp_path)
    paths.raw_zip(2023, 7).unlink()

    monkeypatch.setenv("FLIGHT_DELAY_DATA", str(tmp_path))
    assert main(["status"]) == 1
    assert "2023_7" in capsys.readouterr().out


def test_status_survives_a_missing_lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLIGHT_DELAY_DATA", str(tmp_path / "nope"))
    assert main(["status"]) == 1
