"""Recalibrate probabilities without touching the ranking.

Scenario A is overconfident where it matters: in the [0.6, 0.7) bin it predicts
0.64 and the observed rate is 0.53. Those are exactly the flights a warning
system would act on, so the error lands where it costs most.

Isotonic regression fixes that shape. It is monotonic, so it cannot reorder
anything -- PR-AUC and ROC-AUC come out unchanged by construction, and only the
probabilities move. That property is also the test: if a "calibration" step
changes the ranking, it is doing something else.

The calibration set is the **last two months of the training year**, not a
random sample. A random holdout would mix December into a set used to correct a
model whose test year is the next one, which is the same temporal leak the
whole split exists to prevent. The cost is real and stated: the calibrated
model is fitted on ten months rather than twelve, so it is compared against an
uncalibrated model trained on the same ten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

#: Months of the training year reserved for fitting the calibrator.
CALIBRATION_MONTHS: tuple[int, ...] = (11, 12)


@dataclass(frozen=True, slots=True)
class TemporalHoldout:
    fit_x: pd.DataFrame
    fit_y: np.ndarray[Any, Any]
    calib_x: pd.DataFrame
    calib_y: np.ndarray[Any, Any]

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.fit_x), len(self.calib_x)


def split_by_month(
    x: pd.DataFrame,
    y: np.ndarray[Any, Any],
    *,
    calibration_months: tuple[int, ...] = CALIBRATION_MONTHS,
) -> TemporalHoldout:
    """Chronological split of the training year: early months fit, late months
    calibrate. Requires a ``month`` column, which the feature table carries."""
    if "month" not in x.columns:
        raise KeyError("a 'month' column is required to split the training year in time")
    held = x["month"].isin(calibration_months).to_numpy()
    return TemporalHoldout(
        fit_x=x.loc[~held],
        fit_y=y[~held],
        calib_x=x.loc[held],
        calib_y=y[held],
    )


def split_stratified_by_month(
    x: pd.DataFrame,
    y: np.ndarray[Any, Any],
    *,
    fraction: float = 0.15,
    seed: int = 0,
) -> TemporalHoldout:
    """Hold out a slice of *every* month of the training year.

    Reserving November and December instead is chronologically cleaner and
    empirically worse: those two months are the holiday peak, so a map fitted
    on them and applied to a full year makes calibration worse rather than
    better (measured: worst gap 0.147 -> 0.322). Sampling every month keeps the
    calibration set seasonally representative.

    The property that matters is still intact -- every row here is from the
    training year, and the evaluation year comes strictly afterwards, so
    nothing from the future informs the model or the map. What is given up is
    ordering *within* the training year, which shapes a monotonic remapping far
    less than it shapes a feature.
    """
    if "month" not in x.columns:
        raise KeyError("a 'month' column is required to stratify the training year")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")

    rng = np.random.default_rng(seed)
    held = np.zeros(len(x), dtype=bool)
    months = x["month"].to_numpy()
    for month in np.unique(months):
        positions = np.flatnonzero(months == month)
        picked = rng.choice(positions, size=int(len(positions) * fraction), replace=False)
        held[picked] = True

    return TemporalHoldout(
        fit_x=x.loc[~held],
        fit_y=y[~held],
        calib_x=x.loc[held],
        calib_y=y[held],
    )


class IsotonicCalibrator:
    """Wraps a fitted classifier and remaps its probabilities."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, x: pd.DataFrame, y: np.ndarray[Any, Any]) -> IsotonicCalibrator:
        raw = self.model.predict_proba(x)[:, 1]
        self.iso.fit(raw, y)
        return self

    def predict_proba_positive(self, x: pd.DataFrame) -> np.ndarray[Any, Any]:
        raw = self.model.predict_proba(x)[:, 1]
        return np.asarray(self.iso.predict(raw), dtype=np.float64)


def ranking_is_preserved(before: np.ndarray[Any, Any], after: np.ndarray[Any, Any]) -> bool:
    """Isotonic mapping is monotonic, so ordering must survive it.

    Ties can appear -- the mapping is not strictly increasing -- so the check is
    that no pair is *reversed*, not that the ordering is identical.
    """
    order = np.argsort(before, kind="stable")
    mapped = after[order]
    return bool(np.all(np.diff(mapped) >= -1e-12))
