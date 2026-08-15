from __future__ import annotations

from pathlib import Path

import pytest

from flight_delay.config import (
    MONTHS,
    TEST_YEAR,
    TRAIN_YEAR,
    YEARS,
    Paths,
    mlflow_artifact_uri,
    mlflow_tracking_uri,
)


def test_data_root_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLIGHT_DELAY_DATA", str(tmp_path))
    assert Paths.from_env().root == tmp_path


def test_data_root_defaults_outside_the_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lake must never resolve next to the source tree: that would put
    gigabytes inside OneDrive and behind the /mnt/c bridge."""
    monkeypatch.delenv("FLIGHT_DELAY_DATA", raising=False)
    root = Paths.from_env().root
    assert root.is_absolute()
    assert Path(__file__).parent.parent not in root.parents


def test_raw_zip_name_matches_the_download_script(tmp_path: Path) -> None:
    # scripts/download_bts.sh writes "${year}_${month}.zip", unpadded month.
    assert Paths(root=tmp_path).raw_zip(2024, 3).name == "2024_3.zip"


def test_split_is_temporal_and_contiguous() -> None:
    assert TRAIN_YEAR < TEST_YEAR
    assert YEARS == (TRAIN_YEAR, TEST_YEAR)
    assert tuple(range(1, 13)) == MONTHS


def test_mlflow_uri_is_a_database_inside_the_lake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MLflow 3.15 put the file store into maintenance mode and refuses to
    open one, so tracking goes to SQLite -- still inside WSL, never OneDrive."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = mlflow_tracking_uri(Paths(root=tmp_path))
    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")
    assert str(tmp_path) in uri


def test_artifacts_land_in_the_lake_not_next_to_the_source(
    tmp_path: Path,
) -> None:
    uri = mlflow_artifact_uri(Paths(root=tmp_path))
    assert uri.startswith("file:")
    assert uri.endswith("mlruns")


def test_explicit_mlflow_uri_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    assert mlflow_tracking_uri(Paths(root=tmp_path)) == "http://localhost:5000"
