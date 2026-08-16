from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from flight_delay.serving import api

VALID = {
    "flight_date": "2024-07-15",
    "carrier": "AA",
    "origin": "JFK",
    "dest": "LAX",
    "scheduled_departure": "0830",
    "scheduled_arrival": "1145",
    "distance_miles": 2475,
    "scheduled_elapsed_minutes": 375,
}


class StubModel:
    """Returns a probability derived from the departure time, so a test can
    tell that the request actually reached the model."""

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = float(frame["dep_minute_of_day"].iloc[0]) / 1440.0
        return np.array([[1 - p, p]])


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    import joblib

    joblib.dump(StubModel(), tmp_path / "model.joblib")
    pd.DataFrame(
        {
            "kind": ["origin", "carrier"],
            "key": ["JFK", "AA"],
            "rate": [0.24, 0.22],
            "flights": [30_000, 100_000],
        }
    ).to_parquet(tmp_path / "priors.parquet", index=False)
    (tmp_path / "metadata.json").write_text(json.dumps({"train_year": 2023, "pr_auc": 0.3426}))

    monkeypatch.setenv("FLIGHT_DELAY_MODEL_DIR", str(tmp_path))
    api.get_bundle.cache_clear()
    yield TestClient(api.create_app())
    api.get_bundle.cache_clear()


def test_health_needs_no_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Liveness must not depend on the artifact, or a missing model turns into
    a container that never reports ready."""
    monkeypatch.setenv("FLIGHT_DELAY_MODEL_DIR", str(tmp_path / "nothing"))
    api.get_bundle.cache_clear()
    assert TestClient(api.create_app()).get("/health").json() == {"status": "ok"}


def test_predict_returns_a_probability_and_a_decision(client: TestClient) -> None:
    body = client.post("/predict", json=VALID).json()
    # 0830 -> 510 minutes -> 510/1440
    assert body["delay_probability"] == pytest.approx(0.3542, abs=1e-4)
    assert body["warn"] is True
    assert body["threshold"] == 0.25


def test_the_threshold_is_the_caller_s_choice(client: TestClient) -> None:
    """It depends on what a missed delay costs against a false alarm, which is
    a business input, so it is a request parameter rather than a constant."""
    high = client.post("/predict", json={**VALID, "threshold": 0.9}).json()
    low = client.post("/predict", json={**VALID, "threshold": 0.1}).json()
    assert high["warn"] is False
    assert low["warn"] is True
    assert high["delay_probability"] == low["delay_probability"]


def test_the_response_carries_the_base_rate_for_context(client: TestClient) -> None:
    """'38%' means nothing to a passenger without 'the average flight is 21%'."""
    body = client.post("/predict", json=VALID).json()
    assert body["base_rate"] == pytest.approx(0.2056)
    assert body["times_base_rate"] == pytest.approx(body["delay_probability"] / 0.2056, abs=0.01)


def test_the_response_says_which_model_answered(client: TestClient) -> None:
    body = client.post("/predict", json=VALID).json()
    assert body["model_trained_on"] == "2023"
    assert body["model_pr_auc"] == pytest.approx(0.3426)


def test_a_missing_inbound_is_reported_not_hidden(client: TestClient) -> None:
    without = client.post("/predict", json=VALID).json()
    with_inbound = client.post("/predict", json={**VALID, "inbound_delay_minutes": 45}).json()
    assert without["inbound_known"] is False
    assert with_inbound["inbound_known"] is True


def test_lowercase_codes_are_accepted(client: TestClient) -> None:
    assert client.post("/predict", json={**VALID, "origin": "jfk"}).status_code == 200


@pytest.mark.parametrize(
    "override",
    [
        {"scheduled_departure": "8:30"},
        {"scheduled_departure": "830"},
        {"distance_miles": 0},
        {"distance_miles": -5},
        {"threshold": 1.5},
        {"flight_date": "not-a-date"},
        {"carrier": ""},
    ],
)
def test_malformed_requests_are_rejected(client: TestClient, override: dict[str, Any]) -> None:
    assert client.post("/predict", json={**VALID, **override}).status_code == 422


def test_an_impossible_clock_is_a_422_not_a_500(client: TestClient) -> None:
    """'2570' passes the four-digit pattern and is still not a time."""
    response = client.post("/predict", json={**VALID, "scheduled_departure": "2570"})
    assert response.status_code == 422


def test_model_endpoint_states_what_the_score_means(client: TestClient) -> None:
    body = client.get("/model").json()
    assert body["scenario"] == "A_pre_departure"
    assert "dep_minute_of_day" in body["features"]
    assert "not a deployable warning system" in body["note"]


def test_the_model_endpoint_never_advertises_post_departure_features(
    client: TestClient,
) -> None:
    """Serving scenario B would score beautifully and answer a question nobody
    asks: by then the aircraft has already left."""
    features = client.get("/model").json()["features"]
    assert "dep_delay" not in features
    assert "taxi_out" not in features
