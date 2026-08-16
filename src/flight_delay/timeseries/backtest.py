"""Rolling-origin backtesting against a seasonal baseline.

Cross-validation with random folds is meaningless here: it trains on Wednesday
and tests on the Tuesday before it. The origin moves forward instead, the model
only ever sees the past, and it is refitted as the origin advances -- which is
also what a deployed forecaster would do.

The scale is MASE, mean absolute error divided by the error of the seasonal
naive forecast on the training data. It is the metric that makes "is this worth
deploying" answerable: **MASE below 1 beats last week's value, above 1 does
not**, and a raw MAE of 0.03 on a rate that lives between 0.1 and 0.4 tells you
nothing on its own.

Weekly seasonality dominates this series, so the seasonal naive is a genuinely
hard baseline rather than a strawman.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from flight_delay.timeseries.series import SEASON


@dataclass(frozen=True, slots=True)
class ForecastScore:
    name: str
    horizon: int
    mae: float
    rmse: float
    mase: float
    n: int

    @property
    def beats_seasonal_naive(self) -> bool:
        return self.mase < 1.0

    def as_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "mase": self.mase, "n": float(self.n)}


def seasonal_naive_scale(y_train: np.ndarray[Any, Any], season: int = SEASON) -> float:
    """Mean absolute change over one season, in-sample. The MASE denominator."""
    if y_train.size <= season:
        raise ValueError("training series is shorter than one season")
    scale = float(np.mean(np.abs(y_train[season:] - y_train[:-season])))
    if scale == 0.0:
        raise ValueError("seasonal naive error is zero; MASE is undefined")
    return scale


def score(
    name: str,
    horizon: int,
    y_true: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any],
    scale: float,
) -> ForecastScore:
    errors = np.abs(y_true - y_pred)
    return ForecastScore(
        name=name,
        horizon=horizon,
        mae=float(errors.mean()),
        rmse=float(np.sqrt(((y_true - y_pred) ** 2).mean())),
        mase=float(errors.mean() / scale),
        n=int(y_true.size),
    )


def build_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        early_stopping=False,
        random_state=0,
    )


@dataclass(frozen=True, slots=True)
class BacktestResult:
    predictions: pd.DataFrame
    scores: list[ForecastScore]
    refits: int


def rolling_origin_backtest(
    frame: pd.DataFrame,
    features: list[str],
    *,
    horizon: int,
    first_test_date: pd.Timestamp,
    target: str = "rate",
    refit_every: str = "MS",
) -> BacktestResult:
    """Walk forward through the test period, refitting as the origin moves.

    ``refit_every`` groups the test period into blocks (month starts by
    default). For each block the model is fitted on everything strictly before
    it, then predicts every day inside it. Refitting once per day would be more
    faithful and costs 30x more for a result that does not change the
    conclusion; the block size is stated rather than hidden.
    """
    work = frame.dropna(subset=[*features, target]).sort_values("date").reset_index(drop=True)
    test_mask = work["date"] >= first_test_date
    if not test_mask.any():
        raise ValueError(f"no rows on or after {first_test_date}")

    blocks = work.loc[test_mask, "date"].dt.to_period(refit_every[0]).unique()
    chunks: list[pd.DataFrame] = []
    refits = 0

    for block in blocks:
        block_start = block.start_time
        train = work[work["date"] < block_start]
        test = work[(work["date"] >= block_start) & (work["date"] <= block.end_time)]
        if train.empty or test.empty:
            continue

        model = build_regressor()
        model.fit(train[features], train[target])
        refits += 1

        predicted = test.copy()
        predicted["model"] = model.predict(test[features])
        chunks.append(predicted)

    predictions = pd.concat(chunks, ignore_index=True)

    # The MASE scale comes from the data available before the test period
    # begins, never from the test period itself.
    history = work.loc[work["date"] < first_test_date, target].to_numpy()
    scale = seasonal_naive_scale(history)

    y_true = predictions[target].to_numpy()
    scores = [score("gradient boosting", horizon, y_true, predictions["model"].to_numpy(), scale)]

    for label, column in (
        ("naive (h days ago)", f"lag_{horizon}"),
        ("seasonal naive (same weekday)", "lag_7"),
        ("rolling 28-day mean", "roll_mean_28"),
    ):
        if column in predictions.columns and predictions[column].notna().all():
            scores.append(score(label, horizon, y_true, predictions[column].to_numpy(), scale))

    return BacktestResult(predictions=predictions, scores=scores, refits=refits)


def format_scores(scores: list[ForecastScore]) -> str:
    lines = [f"  {'forecaster':32s} {'MAE':>8s} {'RMSE':>8s} {'MASE':>7s}  beats naive"]
    lines.append("  " + "-" * 68)
    for s in sorted(scores, key=lambda s: s.mase):
        lines.append(
            f"  {s.name:32s} {s.mae:8.5f} {s.rmse:8.5f} {s.mase:7.3f}  "
            f"{'yes' if s.beats_seasonal_naive else 'no'}"
        )
    return "\n".join(lines)
