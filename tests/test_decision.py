from __future__ import annotations

import numpy as np
import pytest

from flight_delay.models.decision import (
    best_threshold,
    cost_sensitivity,
    operating_point,
    sweep,
)

# Four flights, two of them late, with probabilities that separate them.
Y = np.array([1.0, 1.0, 0.0, 0.0])
P = np.array([0.9, 0.4, 0.3, 0.1])


def test_counts_add_up_to_the_sample() -> None:
    point = operating_point(Y, P, 0.35)
    assert point.true_positive == 2
    assert point.false_positive == 0
    assert point.false_negative == 0
    assert point.true_negative == 2


def test_threshold_is_inclusive() -> None:
    """A flight predicted at exactly the threshold is flagged; off-by-one here
    silently changes every reported alert rate."""
    assert operating_point(Y, P, 0.4).predicted_positive == 2
    assert operating_point(Y, P, 0.41).predicted_positive == 1


def test_precision_and_recall_use_their_own_denominators() -> None:
    point = operating_point(Y, P, 0.2)
    assert point.precision == pytest.approx(2 / 3)
    assert point.recall == pytest.approx(1.0)
    assert point.alert_rate == pytest.approx(0.75)


def test_no_alerts_gives_zero_precision_rather_than_dividing_by_zero() -> None:
    point = operating_point(Y, P, 0.99)
    assert point.predicted_positive == 0
    assert point.precision == 0.0
    assert point.recall == 0.0
    assert point.f1 == 0.0


def test_a_costlier_miss_pushes_the_threshold_down() -> None:
    """The whole point of the cost sweep: as a missed delay gets more
    expensive, the system should warn more people, not fewer."""
    points = sweep(Y, P, np.round(np.arange(0.05, 1.0, 0.05), 2))
    cheap = best_threshold(points, 1.0)
    expensive = best_threshold(points, 10.0)
    assert expensive.threshold <= cheap.threshold
    assert expensive.recall >= cheap.recall


def test_expected_cost_prices_a_false_positive_at_one() -> None:
    point = operating_point(Y, P, 0.2)  # 2 TP, 1 FP, 0 FN, 1 TN
    assert point.expected_cost(1.0) == pytest.approx(1 / 4)
    assert point.expected_cost(10.0) == pytest.approx(1 / 4)  # no misses to charge for


def test_expected_cost_charges_misses_at_the_given_rate() -> None:
    point = operating_point(Y, P, 0.95)  # 0 TP, 0 FP, 2 FN, 2 TN
    assert point.expected_cost(1.0) == pytest.approx(2 / 4)
    assert point.expected_cost(5.0) == pytest.approx(10 / 4)


def test_sensitivity_returns_one_point_per_ratio() -> None:
    points = sweep(Y, P, np.array([0.2, 0.5, 0.8]))
    result = cost_sensitivity(points, (1.0, 5.0))
    assert set(result) == {1.0, 5.0}
