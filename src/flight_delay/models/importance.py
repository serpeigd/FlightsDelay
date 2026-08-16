"""Which features actually carry the prediction.

Permutation importance rather than the model's own split counts. Impurity-based
importance in a tree ensemble is biased towards high-cardinality features -- it
would hand the crown to ``origin`` (350 levels) over ``inbound_known`` (2)
regardless of what either contributes. Permutation asks a question that means
the same thing for every column: shuffle it on held-out data, and see how much
the metric falls.

It is scored with PR-AUC, the metric the project reports, not accuracy.

Correlated features share credit and can both look unimportant, which is worth
remembering here: ``inbound_delay``, ``inbound_known`` and
``inbound_turnaround_minutes`` describe one underlying fact between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


@dataclass(frozen=True, slots=True)
class FeatureImportance:
    feature: str
    mean_drop: float
    std_drop: float

    @property
    def relative(self) -> float:
        return self.mean_drop


def permutation_importance(
    predict_proba: Any,
    x: pd.DataFrame,
    y: np.ndarray[Any, Any],
    *,
    features: tuple[str, ...],
    repeats: int = 3,
    sample: int | None = 200_000,
    seed: int = 0,
) -> list[FeatureImportance]:
    """Drop in PR-AUC when each feature is shuffled.

    ``sample`` subsets the evaluation set: this runs ``len(features) * repeats``
    full predictions, which over 7M rows would take longer than training did.
    """
    rng = np.random.default_rng(seed)

    if sample is not None and sample < len(x):
        picked = rng.choice(len(x), size=sample, replace=False)
        x = x.iloc[picked]
        y = y[picked]

    baseline = float(average_precision_score(y, predict_proba(x)))

    results: list[FeatureImportance] = []
    for feature in features:
        drops: list[float] = []
        original = x[feature].to_numpy(copy=True)
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[feature] = rng.permutation(original)
            drops.append(baseline - float(average_precision_score(y, predict_proba(shuffled))))
        results.append(FeatureImportance(feature, float(np.mean(drops)), float(np.std(drops))))

    return sorted(results, key=lambda r: r.mean_drop, reverse=True)


def format_importances(items: list[FeatureImportance], baseline: float) -> str:
    lines = [f"  {'feature':30s} {'PR-AUC drop':>12s} {'std':>8s} {'% of total':>11s}"]
    lines.append("  " + "-" * 63)
    for item in items:
        share = 100 * item.mean_drop / baseline if baseline else 0.0
        lines.append(
            f"  {item.feature:30s} {item.mean_drop:12.5f} {item.std_drop:8.5f} {share:10.1f}%"
        )
    return "\n".join(lines)
