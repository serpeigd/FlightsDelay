"""The artifact the API loads: a model, its priors, and what it scored.

Kept together on purpose. A model file without the reference table that feeds
its congestion features is not deployable, and a model without the metrics it
earned invites someone to quote a number it never achieved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from flight_delay.serving.features import PriorRates

MODEL_FILE = "model.joblib"
PRIORS_FILE = "priors.parquet"
ROUTES_FILE = "routes.parquet"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True, slots=True)
class ModelBundle:
    model: Any
    priors: PriorRates
    metadata: dict[str, Any]
    #: One row per flown route: origin, dest, city names, typical distance and
    #: scheduled duration. Lets a caller pick a real route from a list instead
    #: of having to know that JFK to LAX is 2475 miles.
    routes: pd.DataFrame

    @property
    def trained_on(self) -> str:
        return str(self.metadata.get("train_year", "unknown"))

    @property
    def pr_auc(self) -> float | None:
        value = self.metadata.get("pr_auc")
        return float(value) if value is not None else None

    def predict_proba(self, frame: pd.DataFrame) -> float:
        return float(self.model.predict_proba(frame)[0, 1])


def save(
    directory: Path,
    *,
    model: Any,
    priors: pd.DataFrame,
    routes: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    import joblib

    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / MODEL_FILE)
    priors.to_parquet(directory / PRIORS_FILE, index=False)
    routes.to_parquet(directory / ROUTES_FILE, index=False)
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2))


def load(directory: Path) -> ModelBundle:
    import joblib

    if not (directory / MODEL_FILE).is_file():
        raise FileNotFoundError(f"no model at {directory}. Run `flight-delay export-model` first.")
    routes_file = directory / ROUTES_FILE
    return ModelBundle(
        model=joblib.load(directory / MODEL_FILE),
        priors=PriorRates.from_frame(pd.read_parquet(directory / PRIORS_FILE)),
        metadata=json.loads((directory / METADATA_FILE).read_text()),
        routes=pd.read_parquet(routes_file) if routes_file.is_file() else pd.DataFrame(),
    )
