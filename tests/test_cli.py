from __future__ import annotations

from pathlib import Path

import pytest

from flight_delay.cli import COMMANDS, build_parser, main
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


def test_status_says_how_to_build_a_missing_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_complete_lake(tmp_path)
    monkeypatch.setenv("FLIGHT_DELAY_DATA", str(tmp_path))
    main(["status"])

    out = capsys.readouterr().out
    assert "flight-delay curate" in out
    assert "flight-delay features" in out


def test_status_survives_a_missing_lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLIGHT_DELAY_DATA", str(tmp_path / "nope"))
    assert main(["status"]) == 1


@pytest.mark.parametrize("command", [name for name, _ in COMMANDS])
def test_every_advertised_command_parses(command: str) -> None:
    """The help text and the parser must not drift apart: a command listed in
    COMMANDS but absent from the parser fails only when someone runs it."""
    assert build_parser().parse_args([command]).command == command


def test_every_command_has_help_text() -> None:
    assert all(help_text for _, help_text in COMMANDS)


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_unknown_commands_are_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["definitely-not-a-command"])


def test_extract_accepts_force_and_others_do_not() -> None:
    assert build_parser().parse_args(["extract", "--force"]).force is True
    assert build_parser().parse_args(["extract"]).force is False
    with pytest.raises(SystemExit):
        build_parser().parse_args(["curate", "--force"])


def test_the_pipeline_order_is_documented_in_the_module_docstring() -> None:
    """The docstring is the first thing a reader sees; if a command is missing
    from it the pipeline cannot be followed."""
    import flight_delay.cli as cli

    assert cli.__doc__ is not None
    for name, _ in COMMANDS:
        assert name in cli.__doc__, name
