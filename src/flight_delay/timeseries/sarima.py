"""The classical baseline: seasonal ARIMA on the national series.

Gradient boosting on lags and calendar features is the modern default, and it
is not the tool a statistician reaches for first when handed a daily series
with weekly seasonality. Without SARIMA in the comparison, "the model beats the
seasonal naive" leaves the obvious follow-up unanswered.

It is scored inside the **same** rolling-origin backtest, against the **same**
MASE scale, over the **same** test period, so the numbers sit in one table
rather than needing a caveat.

Order selection happens once, on the training year only, by AIC. Re-selecting
at every refit would be more thorough and would also let the test period
influence the model class, which is the thing the whole protocol exists to
prevent.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from flight_delay.timeseries.backtest import ForecastScore, score, seasonal_naive_scale
from flight_delay.timeseries.series import SEASON

#: Small deliberate grid. A wider search would overfit the selection itself on
#: 365 observations, and the point here is a fair baseline rather than the best
#: possible SARIMA.
ORDERS: tuple[tuple[int, int, int], ...] = ((1, 0, 0), (1, 0, 1), (2, 0, 1), (1, 1, 1))
SEASONAL_ORDERS: tuple[tuple[int, int, int], ...] = ((1, 0, 0), (1, 0, 1), (0, 1, 1))


@dataclass(frozen=True, slots=True)
class Selection:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    aic: float

    @property
    def label(self) -> str:
        p, d, q = self.order
        sp, sd, sq, m = self.seasonal_order
        return f"SARIMA({p},{d},{q})({sp},{sd},{sq})[{m}]"


def _fit(values: np.ndarray[Any, Any], order: Any, seasonal: Any) -> Any:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        # Convergence chatter on a short series is noise, not signal.
        warnings.simplefilter("ignore")
        return SARIMAX(
            values,
            order=order,
            seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)


def select_order(train: np.ndarray[Any, Any], season: int = SEASON) -> Selection:
    """Lowest AIC over the grid, fitted on the training year only."""
    best: Selection | None = None
    for order in ORDERS:
        for seasonal in SEASONAL_ORDERS:
            full_seasonal = (*seasonal, season)
            try:
                aic = float(_fit(train, order, full_seasonal).aic)
            except Exception:
                continue
            if not np.isfinite(aic):
                continue
            if best is None or aic < best.aic:
                best = Selection(order, full_seasonal, aic)

    if best is None:
        raise RuntimeError("no SARIMA order in the grid converged")
    return best


def backtest(
    frame: pd.DataFrame,
    *,
    horizon: int,
    first_test_date: pd.Timestamp,
    target: str = "rate",
    selection: Selection | None = None,
) -> tuple[ForecastScore, Selection, pd.DataFrame]:
    """Rolling-origin SARIMA forecast, refitted monthly like the other models.

    Refitting monthly rather than daily matches what the gradient boosting
    backtest does, so the comparison is between model classes and not between
    refit schedules.
    """
    ordered = frame.sort_values("date").reset_index(drop=True)
    history = ordered.loc[ordered["date"] < first_test_date, target].to_numpy(dtype=float)
    if history.size == 0:
        raise ValueError(f"no observations before {first_test_date}")

    chosen = selection or select_order(history)
    scale = seasonal_naive_scale(history)

    test = ordered[ordered["date"] >= first_test_date]
    predictions: list[dict[str, Any]] = []

    for period, block in test.groupby(test["date"].dt.to_period("M"), sort=True):
        cutoff = period.start_time
        train = ordered.loc[ordered["date"] < cutoff, target].to_numpy(dtype=float)
        fitted = _fit(train, chosen.order, chosen.seasonal_order)

        # One forecast path per block, read at the requested horizon: index
        # h-1 is the first day this origin could be asked about.
        steps = len(block) + horizon
        path = np.asarray(fitted.forecast(steps=steps), dtype=float)

        for offset, (_, row) in enumerate(block.iterrows()):
            predictions.append(
                {
                    "date": row["date"],
                    "actual": float(row[target]),
                    "model": float(path[offset + horizon - 1]),
                }
            )

    out = pd.DataFrame(predictions)
    result = score(chosen.label, horizon, out["actual"].to_numpy(), out["model"].to_numpy(), scale)
    return result, chosen, out
