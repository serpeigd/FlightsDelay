"""From a probability to a decision.

A PR-AUC is not an answer to "should we warn this passenger". That needs a
threshold, and a threshold needs a cost: how much worse is failing to warn
someone whose flight is late than warning someone whose flight is fine?

Nothing here invents that ratio. It is an input, the optimum is reported across
a range of plausible values, and the sensitivity is shown -- because the
honest finding is usually that the recommended threshold moves a lot with an
assumption nobody has measured.

0.5 is never the answer under imbalance. With a 20.8% positive rate, a model
that is calibrated will rarely exceed 0.5 at all, so a 0.5 cut-off warns almost
nobody.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    threshold: float
    predicted_positive: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        flagged = self.true_positive + self.false_positive
        return self.true_positive / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def alert_rate(self) -> float:
        total = self.predicted_positive + self.true_negative + self.false_negative
        return self.predicted_positive / total if total else 0.0

    def expected_cost(self, false_negative_cost: float) -> float:
        """Cost per flight, with a false positive costing 1 by definition."""
        n = self.true_positive + self.false_positive + self.false_negative + self.true_negative
        if not n:
            return 0.0
        return (self.false_positive + false_negative_cost * self.false_negative) / n


def operating_point(y_true: Floats, y_prob: Floats, threshold: float) -> OperatingPoint:
    flagged = y_prob >= threshold
    positive = y_true == 1
    return OperatingPoint(
        threshold=threshold,
        predicted_positive=int(flagged.sum()),
        true_positive=int((flagged & positive).sum()),
        false_positive=int((flagged & ~positive).sum()),
        false_negative=int((~flagged & positive).sum()),
        true_negative=int((~flagged & ~positive).sum()),
    )


def sweep(y_true: Floats, y_prob: Floats, thresholds: Floats) -> list[OperatingPoint]:
    return [operating_point(y_true, y_prob, float(t)) for t in thresholds]


def best_threshold(points: list[OperatingPoint], false_negative_cost: float) -> OperatingPoint:
    return min(points, key=lambda p: p.expected_cost(false_negative_cost))


def cost_sensitivity(
    points: list[OperatingPoint], ratios: tuple[float, ...]
) -> dict[float, OperatingPoint]:
    """Optimal operating point for each assumed cost ratio.

    Reported as a table rather than a single number: the ratio is a business
    input, and showing how much the answer moves with it is more useful than
    picking one and calling it optimal.
    """
    return {ratio: best_threshold(points, ratio) for ratio in ratios}
