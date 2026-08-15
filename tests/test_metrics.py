from __future__ import annotations

import numpy as np
import pytest

from flight_delay.models.metrics import calibration_table, evaluate

RNG = np.random.default_rng(0)


def imbalanced(n: int = 20_000, rate: float = 0.2) -> np.ndarray:
    return (RNG.random(n) < rate).astype(np.float64)


def test_constant_predictor_has_pr_auc_equal_to_the_base_rate() -> None:
    """This is why PR-AUC is the headline: the floor is the positive rate, not
    0.5, so 'better than chance' is a visible bar."""
    y = imbalanced()
    result = evaluate("constant", y, np.full(y.size, 0.2))
    assert result.pr_auc == pytest.approx(y.mean(), abs=0.01)
    assert result.lift_over_base_rate == pytest.approx(1.0, abs=0.05)


def test_constant_predictor_gets_roc_auc_of_one_half_not_an_error() -> None:
    y = imbalanced()
    assert evaluate("constant", y, np.full(y.size, 0.2)).roc_auc == 0.5


def test_a_perfect_ranker_reaches_pr_auc_one() -> None:
    y = imbalanced()
    assert evaluate("oracle", y, y * 0.98 + 0.01).pr_auc == pytest.approx(1.0, abs=1e-6)


def test_ranking_and_calibration_are_independent() -> None:
    """A model can order flights perfectly and still be badly calibrated. If
    the number is shown to a passenger, the second failure is the one that
    matters, so they are reported separately."""
    y = imbalanced()
    perfect_order_bad_probabilities = y * 0.4 + 0.55  # 0.55 vs 0.95, never right

    result = evaluate("miscalibrated oracle", y, perfect_order_bad_probabilities)
    assert result.pr_auc == pytest.approx(1.0, abs=1e-6)
    assert result.max_calibration_gap > 0.4


def test_calibration_bins_are_equal_width_and_skip_empty_ones() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    bins = calibration_table(y, np.array([0.05, 0.05, 0.95, 0.95]), bins=10)
    assert len(bins) == 2
    assert bins[0].lower == pytest.approx(0.0)
    assert bins[-1].upper == pytest.approx(1.0)


def test_calibration_reports_the_observed_rate_not_the_prediction() -> None:
    y = np.array([1.0, 1.0, 0.0, 0.0])
    (single,) = calibration_table(y, np.full(4, 0.95), bins=10)
    assert single.mean_predicted == pytest.approx(0.95)
    assert single.observed_rate == pytest.approx(0.5)
    assert single.gap == pytest.approx(0.45)


def test_probability_one_lands_in_the_top_bin() -> None:
    y = np.array([1.0, 1.0])
    (single,) = calibration_table(y, np.array([1.0, 1.0]), bins=10)
    assert single.upper == pytest.approx(1.0)
    assert single.count == 2


def test_evaluation_records_the_sample_it_was_scored_on() -> None:
    y = imbalanced(5_000)
    result = evaluate("m", y, RNG.random(5_000))
    assert result.n == 5_000
    assert result.positive_rate == pytest.approx(y.mean())
