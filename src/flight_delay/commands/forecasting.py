"""Rolling-origin backtest of the daily delay rate.

Produces every figure in ``docs/timeseries.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from flight_delay.commands._common import duckdb_connection, log, start_experiment, write_result
from flight_delay.config import Paths
from flight_delay.timeseries.backtest import format_scores, rolling_origin_backtest
from flight_delay.timeseries.sarima import backtest as sarima_backtest
from flight_delay.timeseries.series import (
    SEASON,
    add_calendar_features,
    add_lags,
    airport_series,
    national_series,
)

EXPERIMENT = "flight-delay-timeseries"
FIRST_TEST = pd.Timestamp("2024-01-01")
HORIZONS = (1, 7)
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _describe(national: pd.DataFrame) -> None:
    log(
        f"national series: {len(national)} days, "
        f"{national['date'].min().date()} to {national['date'].max().date()}"
    )
    log(
        f"  delay rate  min {national['rate'].min():.3f}  "
        f"median {national['rate'].median():.3f}  max {national['rate'].max():.3f}"
    )

    train = national[national["date"] < FIRST_TEST]
    log("\n== mean delay rate by weekday (2023) ==")
    for dow, name in enumerate(WEEKDAYS):
        day = train[train["day_of_week"] == dow]
        if len(day):
            log(f"  {name}  {day['rate'].mean():.4f}")

    log("\n== the ten worst days of 2023 ==")
    for _, row in train.nlargest(10, "rate").iterrows():
        log(
            f"  {row['date'].date()}  rate={row['rate']:.3f}  "
            f"flights={int(row['flights']):,}  "
            f"days_from_holiday={int(row['days_from_holiday'])}"
        )


def cmd_forecast(paths: Paths) -> int:
    import mlflow

    start_experiment(paths, EXPERIMENT)
    summary: dict[str, Any] = {}

    with duckdb_connection(paths) as con:
        national = add_calendar_features(national_series(con, paths))
        airports = add_calendar_features(airport_series(con, paths, top_n=10))

    _describe(national)

    for label, frame, group in (("national", national, None), ("airports", airports, "origin")):
        summary[label] = {}
        for horizon in HORIZONS:
            featured, features = add_lags(frame, horizon=horizon, group=group)
            result = rolling_origin_backtest(
                featured, features, horizon=horizon, first_test_date=FIRST_TEST
            )
            log(
                f"\n== {label}, horizon {horizon} day(s) "
                f"({result.refits} refits, {len(result.predictions):,} forecasts) =="
            )
            log(format_scores(result.scores))

            summary[label][f"h{horizon}"] = [
                {"name": s.name, **{k: round(v, 6) for k, v in s.as_dict().items()}}
                for s in result.scores
            ]

            if group is None:
                # The classical baseline, inside the same backtest and scaled
                # by the same MASE denominator so it lands in the same table.
                log("\n  fitting SARIMA on the same protocol...")
                sarima, chosen, _ = sarima_backtest(
                    frame, horizon=horizon, first_test_date=FIRST_TEST
                )
                log(f"  {chosen.label}  (AIC {chosen.aic:.0f} on 2023)")
                log(
                    f"  MAE {sarima.mae:.5f}  MASE {sarima.mase:.3f}  "
                    f"{'beats' if sarima.beats_seasonal_naive else 'LOSES to'} seasonal naive"
                )
                summary[label][f"h{horizon}"].append(
                    {"name": chosen.label, **{k: round(v, 6) for k, v in sarima.as_dict().items()}}
                )

                # The series itself, not just scores about it. Without this the
                # dashboard reports MASE for a curve nobody can see.
                summary.setdefault("national_curve", {})[f"h{horizon}"] = _curve(result.predictions)

            if group:
                log("\n  MASE per airport (gradient boosting):")
                per_airport = _per_group_mase(result.predictions, frame, group)
                summary[label][f"h{horizon}_per_airport"] = per_airport
                for name, mase in sorted(per_airport.items(), key=lambda kv: kv[1]):
                    verdict = "beats" if mase < 1 else "LOSES to"
                    log(f"    {name}  MASE {mase:.3f}  {verdict} seasonal naive")

            with mlflow.start_run(run_name=f"{label} | horizon {horizon}"):
                mlflow.log_params(
                    {
                        "series": label,
                        "horizon": horizon,
                        "refits": result.refits,
                        "features": len(features),
                    }
                )
                for s in result.scores:
                    safe = s.name.replace(" ", "_").replace("(", "").replace(")", "")
                    mlflow.log_metrics({f"{safe}_{k}": v for k, v in s.as_dict().items()})

    write_result(paths, "timeseries.json", summary)
    return 0


def _curve(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Observed and forecast rate per day, for plotting.

    Column arrays rather than ``itertuples``: the latter hands back a union of
    every dtype in the frame, which is unusable without casting each field.
    """
    ordered = predictions.sort_values("date")
    dates = ordered["date"].dt.strftime("%Y-%m-%d").tolist()
    actual = ordered["rate"].to_numpy(dtype=float)
    model = ordered["model"].to_numpy(dtype=float)
    return [
        {"date": d, "actual": round(a, 5), "model": round(m, 5)}
        for d, a, m in zip(dates, actual, model, strict=True)
    ]


def _per_group_mase(
    predictions: pd.DataFrame, history: pd.DataFrame, group: str
) -> dict[str, float]:
    """MASE per series, each scaled by its *own* pre-test seasonal error.

    Pooling the scale across airports would let a volatile airport flatter a
    stable one, which is exactly what MASE exists to prevent.
    """
    out: dict[str, float] = {}
    for name, block in predictions.groupby(group):
        past = history[(history[group] == name) & (history["date"] < FIRST_TEST)]
        values = past["rate"].to_numpy()
        scale = float(np.mean(np.abs(values[SEASON:] - values[:-SEASON])))
        errors = np.abs(block["rate"].to_numpy() - block["model"].to_numpy())
        out[str(name)] = float(errors.mean() / scale)
    return out
