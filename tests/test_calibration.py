from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_delay.models.calibration import (
    CALIBRATION_MONTHS,
    IsotonicCalibrator,
    ranking_is_preserved,
    split_by_month,
    split_stratified_by_month,
)


def year_frame(rows_per_month: int = 100) -> tuple[pd.DataFrame, np.ndarray]:
    months = np.repeat(np.arange(1, 13), rows_per_month)
    x = pd.DataFrame({"month": months, "value": np.arange(months.size, dtype=float)})
    y = (np.arange(months.size) % 5 == 0).astype(float)
    return x, y


class ConstantOrderModel:
    """Returns the 'value' column as a probability, so ordering is known."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = (x["value"].to_numpy() % 100) / 100.0
        return np.column_stack([1 - p, p])


def test_month_split_reserves_exactly_the_named_months() -> None:
    x, y = year_frame()
    holdout = split_by_month(x, y)
    assert set(holdout.calib_x["month"].unique()) == set(CALIBRATION_MONTHS)
    assert not set(holdout.fit_x["month"].unique()) & set(CALIBRATION_MONTHS)


def test_stratified_split_touches_every_month() -> None:
    """Reserving whole months leaves the model blind to that season, and the
    test year contains it: measured worst calibration gap 0.147 against 0.100
    for a model that saw a slice of each month."""
    x, y = year_frame()
    holdout = split_stratified_by_month(x, y, fraction=0.2)
    assert set(holdout.calib_x["month"].unique()) == set(range(1, 13))
    assert set(holdout.fit_x["month"].unique()) == set(range(1, 13))


def test_stratified_split_holds_out_the_requested_share() -> None:
    x, y = year_frame(rows_per_month=100)
    holdout = split_stratified_by_month(x, y, fraction=0.15)
    assert len(holdout.calib_x) == pytest.approx(0.15 * len(x), abs=12)
    assert len(holdout.fit_x) + len(holdout.calib_x) == len(x)


def test_stratified_split_is_deterministic() -> None:
    x, y = year_frame()
    first = split_stratified_by_month(x, y, seed=7).calib_x.index
    second = split_stratified_by_month(x, y, seed=7).calib_x.index
    assert list(first) == list(second)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_fractions_are_rejected(fraction: float) -> None:
    x, y = year_frame()
    with pytest.raises(ValueError, match="fraction"):
        split_stratified_by_month(x, y, fraction=fraction)


def test_splitting_without_a_month_column_fails_loudly() -> None:
    x = pd.DataFrame({"value": [1.0, 2.0]})
    y = np.array([0.0, 1.0])
    for splitter in (split_by_month, split_stratified_by_month):
        with pytest.raises(KeyError, match="month"):
            splitter(x, y)


def test_isotonic_cannot_reorder_predictions() -> None:
    """The property that makes calibration safe: a monotonic map leaves PR-AUC
    alone. If a 'calibration' step changes the ranking, it is doing something
    else."""
    x, y = year_frame()
    model = ConstantOrderModel()
    calibrator = IsotonicCalibrator(model).fit(x, y)

    raw = model.predict_proba(x)[:, 1]
    mapped = calibrator.predict_proba_positive(x)
    assert ranking_is_preserved(raw, mapped)


def test_ranking_check_catches_a_reversal() -> None:
    before = np.array([0.1, 0.5, 0.9])
    assert ranking_is_preserved(before, np.array([0.2, 0.4, 0.8]))
    assert not ranking_is_preserved(before, np.array([0.9, 0.5, 0.1]))


def test_ranking_check_tolerates_ties() -> None:
    """Isotonic output is a step function, so it creates ties. Ties are not
    reversals."""
    before = np.array([0.1, 0.2, 0.3])
    assert ranking_is_preserved(before, np.array([0.25, 0.25, 0.25]))


def test_calibrated_probabilities_stay_in_range() -> None:
    x, y = year_frame()
    mapped = IsotonicCalibrator(ConstantOrderModel()).fit(x, y).predict_proba_positive(x)
    assert mapped.min() >= 0.0
    assert mapped.max() <= 1.0
