"""Paths and constants for the flight-delay project.

The repo lives on a Windows/OneDrive path but every byte of data lives inside
WSL. Measured with ``scripts/bench_fs.sh``: writing 300 small files takes 19 ms
on the WSL filesystem and 851-1470 ms through ``/mnt/c``. Nothing heavy is
allowed to cross that bridge, so every path below is resolved against the Linux
home directory, never against the location of this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

#: Training year and held-out year. The split is temporal, never random: with
#: seasonal effects and propagated delays, a random k-fold leaks the future
#: into the past.
TRAIN_YEAR: Final = 2023
TEST_YEAR: Final = 2024
YEARS: Final = (TRAIN_YEAR, TEST_YEAR)
MONTHS: Final = tuple(range(1, 13))

#: The label: arrival 15+ minutes late. Null for cancelled and diverted
#: flights (2.2% of rows, measured on 2023-01), which are excluded from the
#: classification dataset and reported separately.
LABEL: Final = "ArrDel15"

_ENV_DATA_ROOT: Final = "FLIGHT_DELAY_DATA"
_DEFAULT_DATA_ROOT: Final = "~/data/flight-delay"


def _resolve_data_root() -> Path:
    raw = os.environ.get(_ENV_DATA_ROOT, _DEFAULT_DATA_ROOT)
    return Path(raw).expanduser()


@dataclass(frozen=True, slots=True)
class Paths:
    """Filesystem layout of the data lake.

    Override the root with the ``FLIGHT_DELAY_DATA`` environment variable;
    tests use it to point at a temporary directory.
    """

    root: Path

    @classmethod
    def from_env(cls) -> Paths:
        return cls(root=_resolve_data_root())

    @property
    def raw(self) -> Path:
        """Downloaded BTS monthly ZIPs, one per year-month."""
        return self.root / "raw"

    @property
    def reference(self) -> Path:
        """Third-party reference data (OPTD airport metadata)."""
        return self.root / "reference"

    @property
    def lake(self) -> Path:
        """Delta/Parquet tables produced by the ingestion pipeline."""
        return self.root / "lake"

    @property
    def mlruns(self) -> Path:
        """MLflow tracking store. Kept out of the repo so OneDrive never
        syncs experiment artifacts."""
        return self.root / "mlruns"

    @property
    def bench(self) -> Path:
        """Benchmark results: JSON emitted by the Spark/DuckDB comparisons."""
        return self.root / "bench"

    def raw_zip(self, year: int, month: int) -> Path:
        """Path of one downloaded month, matching ``scripts/download_bts.sh``."""
        return self.raw / f"{year}_{month}.zip"

    def table(self, name: str) -> Path:
        return self.lake / name


def mlflow_tracking_uri(paths: Paths | None = None) -> str:
    """Tracking URI honouring ``MLFLOW_TRACKING_URI`` if the caller set one."""
    explicit = os.environ.get("MLFLOW_TRACKING_URI")
    if explicit:
        return explicit
    target = (paths or Paths.from_env()).mlruns
    return f"file:{target}"
