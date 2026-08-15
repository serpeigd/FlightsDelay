"""Evaluation for a 20.6%-positive label.

Accuracy is excluded on purpose: predicting "never late" scores 79.4% and is
worthless. So is a bare ROC-AUC headline -- its false-positive rate divides by
the large negative class, which keeps it flattering under imbalance. PR-AUC
uses the denominator that matters: of the flights flagged, how many were
actually late.

Discrimination and calibration are reported separately because they are
different properties. A model can rank flights perfectly and still say "70%"
where the truth is 30%. If a probability is shown to a passenger, the second
number is the one that has to be right, so Brier score and a calibration table
are first-class output rather than an appendix.

Every metric is also computed for the baselines, since a score only means
something next to what it beats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

Floats = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float

    @property
    def gap(self) -> float:
        return self.mean_predicted - self.observed_rate


@dataclass(frozen=True, slots=True)
class Evaluation:
    name: str
    pr_auc: float
    roc_auc: float
    brier: float
    log_loss: float
    positive_rate: float
    n: int
    calibration: tuple[CalibrationBin, ...] = field(default=(), compare=False)

    @property
    def lift_over_base_rate(self) -> float:
        """PR-AUC relative to a constant predictor, whose PR-AUC is the
        positive rate. Below 1.0 means the model adds nothing."""
        return self.pr_auc / self.positive_rate if self.positive_rate else float("nan")

    @property
    def max_calibration_gap(self) -> float:
        return max((abs(b.gap) for b in self.calibration), default=0.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "lift_over_base_rate": self.lift_over_base_rate,
            "max_calibration_gap": self.max_calibration_gap,
        }


def calibration_table(
    y_true: Floats, y_prob: Floats, *, bins: int = 10
) -> tuple[CalibrationBin, ...]:
    """Observed frequency against predicted probability, in equal-width bins.

    Equal-width rather than equal-count so that the sparsely populated
    high-probability region stays visible instead of being merged away -- that
    is where an overconfident model does its damage.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    # right=False keeps 0.0 in the first bin; the top edge is folded back in.
    index = np.digitize(y_prob, edges[1:-1], right=False)

    out: list[CalibrationBin] = []
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if not count:
            continue
        out.append(
            CalibrationBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                count=count,
                mean_predicted=float(y_prob[mask].mean()),
                observed_rate=float(y_true[mask].mean()),
            )
        )
    return tuple(out)


def evaluate(name: str, y_true: Floats, y_prob: Floats, *, bins: int = 10) -> Evaluation:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-15, 1 - 1e-15)

    constant = float(np.unique(y_prob).size == 1)
    return Evaluation(
        name=name,
        pr_auc=float(average_precision_score(y_true, y_prob)),
        # A constant predictor has no ordering, so ROC-AUC is 0.5 by definition
        # rather than an error.
        roc_auc=0.5 if constant else float(roc_auc_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        log_loss=float(log_loss(y_true, y_prob, labels=[0, 1])),
        positive_rate=float(y_true.mean()),
        n=int(y_true.size),
        calibration=calibration_table(y_true, y_prob, bins=bins),
    )


def format_table(evaluations: list[Evaluation]) -> str:
    header = (
        f"  {'model':32s} {'PR-AUC':>8s} {'lift':>6s} {'ROC-AUC':>8s} {'Brier':>8s} {'cal.gap':>8s}"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for e in evaluations:
        lines.append(
            f"  {e.name:32s} {e.pr_auc:8.4f} {e.lift_over_base_rate:6.2f} "
            f"{e.roc_auc:8.4f} {e.brier:8.4f} {e.max_calibration_gap:8.4f}"
        )
    return "\n".join(lines)
