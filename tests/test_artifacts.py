from __future__ import annotations

import json
from pathlib import Path

import pytest

from flight_delay.commands.publish import BUNDLE_FILES, RESULT_FILES, repo_artifacts
from flight_delay.serving import artifacts


def test_the_committed_copy_is_used_when_there_is_no_lake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This is what makes the dashboard deployable somewhere with no data
    lake at all."""
    monkeypatch.setenv("FLIGHT_DELAY_DATA", str(tmp_path / "nothing"))
    assert artifacts.source_label("classification.json") == "committed artifact"
    assert artifacts.load_result("classification.json") is not None


def test_the_live_lake_wins_when_it_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preferring the committed copy would let someone stare at stale numbers
    while believing they had just regenerated them."""
    bench = tmp_path / "bench"
    bench.mkdir(parents=True)
    (bench / "classification.json").write_text(json.dumps({"marker": "from the lake"}))

    monkeypatch.setenv("FLIGHT_DELAY_DATA", str(tmp_path))
    assert artifacts.source_label("classification.json") == "live data lake"
    result = artifacts.load_result("classification.json")
    assert result == {"marker": "from the lake"}


def test_a_file_in_neither_place_is_none_not_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLIGHT_DELAY_DATA", str(tmp_path))
    assert artifacts.load_result("does-not-exist.json") is None
    assert artifacts.source_label("does-not-exist.json") == "missing"


def test_the_model_falls_back_to_the_committed_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLIGHT_DELAY_DATA", str(tmp_path / "nothing"))
    directory = artifacts.model_directory()
    assert directory is not None
    assert (directory / "model.joblib").is_file()


def test_every_published_result_is_actually_committed() -> None:
    """The dashboard reads these by name; a missing one shows an empty page on
    a deployment nobody can debug from a laptop."""
    results = repo_artifacts() / "results"
    for name in RESULT_FILES:
        assert (results / name).is_file(), name


def test_the_committed_bundle_is_complete() -> None:
    model = repo_artifacts() / "model"
    for name in BUNDLE_FILES:
        assert (model / name).is_file(), name


def test_committed_artifacts_stay_small_enough_to_version() -> None:
    """A model that quietly grows to 50 MB should fail here rather than in
    somebody's clone."""
    total = sum(p.stat().st_size for p in repo_artifacts().rglob("*") if p.is_file())
    assert total < 5_000_000, f"artifacts/ is {total / 1e6:.1f} MB"


def test_the_committed_metadata_reports_a_real_score() -> None:
    metadata = json.loads((repo_artifacts() / "model" / "metadata.json").read_text())
    assert metadata["train_year"] == 2023
    assert metadata["scenario"] == "A_pre_departure"
    assert 0.2 < metadata["pr_auc"] < 0.5
