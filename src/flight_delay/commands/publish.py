"""Copy the small artifacts the dashboard needs into the repo.

The data lake lives in WSL and is 1.1 GB; none of it belongs in git. But the
dashboard only reads a handful of JSON summaries and a 1 MB model bundle, and
committing those is what lets the app run somewhere that has no lake at all --
Streamlit Community Cloud, a reviewer's laptop, a container.

Copied by a command rather than by hand so the published artifacts cannot
quietly drift from the ones the pipeline produced.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from flight_delay.commands._common import log
from flight_delay.config import Paths

#: Small enough to version, and the only things the dashboard reads.
RESULT_FILES = (
    "classification.json",
    "calibration.json",
    "analysis.json",
    "timeseries.json",
    "layout.json",
    "engines.json",
    "clustering.json",
)
BUNDLE_FILES = ("model.joblib", "priors.parquet", "routes.parquet", "metadata.json")

#: Anything larger than this is a lake artifact that should not be in git.
MAX_FILE_BYTES = 8 * 1024 * 1024


def repo_artifacts() -> Path:
    """``artifacts/`` at the repository root, next to ``src/``."""
    return Path(__file__).resolve().parents[3] / "artifacts"


def cmd_publish_artifacts(paths: Paths) -> int:
    target = repo_artifacts()
    (target / "results").mkdir(parents=True, exist_ok=True)
    (target / "model").mkdir(parents=True, exist_ok=True)

    copied = 0
    total = 0
    missing: list[str] = []

    for name in RESULT_FILES:
        source = paths.bench / name
        if not source.is_file():
            missing.append(name)
            continue
        shutil.copy2(source, target / "results" / name)
        copied += 1
        total += source.stat().st_size

    for name in BUNDLE_FILES:
        source = paths.root / "model" / name
        if not source.is_file():
            missing.append(f"model/{name}")
            continue
        if source.stat().st_size > MAX_FILE_BYTES:
            log(
                f"REFUSING {name}: {source.stat().st_size / 1e6:.1f} MB is too "
                "large to version. Check the model did not grow unexpectedly."
            )
            return 1
        shutil.copy2(source, target / "model" / name)
        copied += 1
        total += source.stat().st_size

    log(f"copied {copied} file(s), {total / 1e6:.2f} MB, into {target}")
    if missing:
        log(f"missing (run the step that produces them): {', '.join(missing)}")
        return 1
    return 0
