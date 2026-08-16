"""Where the dashboard finds its inputs.

Two places, in order. The WSL data lake wins when it exists, so a developer
always sees what the last pipeline run produced. The copy committed under
``artifacts/`` is the fallback, which is what makes the app runnable somewhere
that has no lake — Streamlit Community Cloud, a container, a reviewer's laptop.

Resolution order matters and is the reason this is a module rather than two
lines inline: silently preferring the committed copy would let a developer
stare at stale numbers while believing they had just regenerated them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flight_delay.config import Paths


def _repo_artifacts() -> Path:
    return Path(__file__).resolve().parents[3] / "artifacts"


def result_path(name: str) -> Path | None:
    """A benchmark JSON, preferring the live lake over the committed copy."""
    for candidate in (Paths.from_env().bench / name, _repo_artifacts() / "results" / name):
        if candidate.is_file():
            return candidate
    return None


def load_result(name: str) -> dict[str, Any] | None:
    path = result_path(name)
    if path is None:
        return None
    parsed: dict[str, Any] = json.loads(path.read_text())
    return parsed


def model_directory() -> Path | None:
    """The model bundle, preferring the live lake over the committed copy."""
    for candidate in (Paths.from_env().root / "model", _repo_artifacts() / "model"):
        if (candidate / "model.joblib").is_file():
            return candidate
    return None


def source_label(name: str) -> str:
    """Which of the two a file came from, so the page can say so."""
    path = result_path(name)
    if path is None:
        return "missing"
    return "committed artifact" if _repo_artifacts() in path.parents else "live data lake"
