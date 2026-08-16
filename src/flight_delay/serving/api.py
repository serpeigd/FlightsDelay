"""FastAPI service for the pre-departure model.

The response carries three things rather than one: the probability, the
decision at the caller's chosen threshold, and the base rate. The last is there
because "38%" means nothing to a passenger without "the average flight is 21%".

The threshold is a request parameter, not a constant. Whether to warn depends
on what a missed delay costs against a false alarm, and that is a business
input this project cannot measure -- 0.5 in particular is close to switching
the service off, since a calibrated model rarely exceeds it on a 21%-positive
label.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from flight_delay.config import Paths
from flight_delay.serving import bundle
from flight_delay.serving.features import build_row

#: Overall delay rate in the training year. Context for a probability.
BASE_RATE = 0.2056


class FlightRequest(BaseModel):
    """A scheduled flight, as known roughly two hours before departure."""

    flight_date: date
    carrier: str = Field(min_length=2, max_length=3, examples=["AA"])
    origin: str = Field(min_length=3, max_length=4, examples=["JFK"])
    dest: str = Field(min_length=3, max_length=4, examples=["LAX"])
    scheduled_departure: str = Field(pattern=r"^\d{4}$", examples=["0830"])
    scheduled_arrival: str = Field(pattern=r"^\d{4}$", examples=["1145"])
    distance_miles: float = Field(gt=0, examples=[2475])
    scheduled_elapsed_minutes: float = Field(gt=0, examples=[375])

    #: The inbound aircraft, when the caller happens to know it. Most will not:
    #: 72.8% of training rows had no usable inbound either, so leaving these
    #: null is in-distribution rather than a degraded request.
    inbound_delay_minutes: float | None = None
    inbound_turnaround_minutes: float | None = Field(default=None, ge=0)

    threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    delay_probability: float
    warn: bool
    threshold: float
    base_rate: float
    times_base_rate: float
    inbound_known: bool
    model_trained_on: str
    model_pr_auc: float | None


class ModelInfo(BaseModel):
    trained_on: str
    pr_auc: float | None
    scenario: str
    features: list[str]
    note: str


def bundle_directory() -> Path:
    override = os.environ.get("FLIGHT_DELAY_MODEL_DIR")
    return Path(override) if override else Paths.from_env().root / "model"


@lru_cache(maxsize=1)
def get_bundle() -> bundle.ModelBundle:
    """Loaded once per process; the model is a few MB and stateless."""
    return bundle.load(bundle_directory())


def create_app() -> FastAPI:
    app = FastAPI(
        title="Flight delay prediction",
        version="0.1.0",
        summary="Pre-departure delay probability for US domestic flights.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model", response_model=ModelInfo)
    def model_info(loaded: Annotated[bundle.ModelBundle, Depends(get_bundle)]) -> ModelInfo:
        from flight_delay.features.build import SCENARIO_A_FEATURES

        return ModelInfo(
            trained_on=loaded.trained_on,
            pr_auc=loaded.pr_auc,
            scenario="A_pre_departure",
            features=list(SCENARIO_A_FEATURES),
            note=(
                "Scores reported on 2024, a year the model never saw. PR-AUC "
                "0.34 against a 0.21 base rate: useful for ranking, not a "
                "deployable warning system on its own."
            ),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        request: FlightRequest,
        loaded: Annotated[bundle.ModelBundle, Depends(get_bundle)],
    ) -> PredictionResponse:
        try:
            frame = build_row(
                flight_date=request.flight_date,
                carrier=request.carrier.upper(),
                origin=request.origin.upper(),
                dest=request.dest.upper(),
                scheduled_departure=request.scheduled_departure,
                scheduled_arrival=request.scheduled_arrival,
                distance=request.distance_miles,
                scheduled_elapsed_minutes=request.scheduled_elapsed_minutes,
                priors=loaded.priors,
                inbound_delay_minutes=request.inbound_delay_minutes,
                inbound_turnaround_minutes=request.inbound_turnaround_minutes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        probability = loaded.predict_proba(frame)
        return PredictionResponse(
            delay_probability=round(probability, 4),
            warn=probability >= request.threshold,
            threshold=request.threshold,
            base_rate=BASE_RATE,
            times_base_rate=round(probability / BASE_RATE, 2),
            inbound_known=request.inbound_delay_minutes is not None,
            model_trained_on=loaded.trained_on,
            model_pr_auc=loaded.pr_auc,
        )

    return app


app: Any = create_app()
