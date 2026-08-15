from __future__ import annotations

import numpy as np
import pandas as pd

from flight_delay.features.build import SCENARIO_A_FEATURES, SCENARIO_B_FEATURES
from flight_delay.models.train import (
    MAX_CATEGORIES,
    RARE,
    RareCategoryGrouper,
    Split,
    baseline_predictions,
)


def frame(origins: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"origin": origins, "dest": ["JFK"] * len(origins), "carrier": ["AA"] * len(origins)}
    )


def test_grouper_keeps_the_frequent_and_pools_the_tail() -> None:
    train = frame(["ATL"] * 100 + ["JFK"] * 50 + ["XXX"] * 2 + ["YYY"] * 1)
    grouper = RareCategoryGrouper(max_categories=2).fit(train)

    out = grouper.transform(train)
    assert set(out["origin"].unique()) == {"ATL", "JFK", RARE}


def test_kept_levels_come_from_training_only() -> None:
    """Letting the test frame vote would let 2024 traffic decide which 2023
    airports the model is allowed to see."""
    train = frame(["ATL"] * 100 + ["JFK"] * 50)
    test = frame(["ZZZ"] * 10_000 + ["ATL"] * 1)

    grouper = RareCategoryGrouper(max_categories=2).fit(train)
    out = grouper.transform(test)

    assert set(out["origin"].unique()) <= {"ATL", "JFK", RARE}
    assert (out["origin"] == RARE).sum() == 10_000


def test_unseen_categories_do_not_crash_at_inference() -> None:
    grouper = RareCategoryGrouper(max_categories=5).fit(frame(["ATL", "JFK"]))
    assert grouper.transform(frame(["BRAND_NEW"]))["origin"].iloc[0] == RARE


def test_coverage_reports_what_survived() -> None:
    train = frame(["ATL"] * 90 + ["ZZZ"] * 10)
    grouper = RareCategoryGrouper(max_categories=1).fit(train)
    assert grouper.coverage(train)["origin"] == 0.9


def test_default_stays_under_the_gradient_boosting_limit() -> None:
    """HistGradientBoosting rejects a categorical with more than 255 levels;
    the pooled level needs a slot too."""
    assert MAX_CATEGORIES <= 254


def test_baselines_fall_back_to_the_base_rate_where_the_prior_is_missing() -> None:
    """2023-01 has no previous month. A NaN prior must become the base rate,
    not a NaN probability that breaks scoring."""
    test_x = pd.DataFrame(
        {
            "origin_prior_rate": [0.3, np.nan],
            "carrier_prior_rate": [np.nan, 0.1],
        }
    )
    split = Split(
        train_x=test_x,
        train_y=np.array([1.0, 0.0, 0.0, 0.0]),
        test_x=test_x,
        test_y=np.array([1.0, 0.0]),
    )
    predictions = baseline_predictions(split)

    assert np.isclose(split.base_rate, 0.25)
    for name, values in predictions.items():
        assert not np.isnan(values).any(), name
    assert np.allclose(predictions["baseline: origin prior-month rate"], [0.3, 0.25])
    assert np.allclose(predictions["baseline: carrier prior-month rate"], [0.25, 0.1])


def test_scenario_b_is_scenario_a_plus_the_departure_record() -> None:
    assert set(SCENARIO_A_FEATURES) < set(SCENARIO_B_FEATURES)
    assert set(SCENARIO_B_FEATURES) - set(SCENARIO_A_FEATURES) == {"dep_delay", "taxi_out"}
