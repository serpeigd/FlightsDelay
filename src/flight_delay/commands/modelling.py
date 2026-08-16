"""Training, calibration, feature importance and threshold choice.

Produces every figure in ``docs/modelling.md``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from flight_delay.commands._common import duckdb_connection, log, start_experiment, write_result
from flight_delay.config import Paths
from flight_delay.features.build import SCENARIO_A_FEATURES
from flight_delay.models.calibration import (
    IsotonicCalibrator,
    TemporalHoldout,
    ranking_is_preserved,
    split_by_month,
    split_stratified_by_month,
)
from flight_delay.models.decision import cost_sensitivity, sweep
from flight_delay.models.importance import format_importances, permutation_importance
from flight_delay.models.metrics import Evaluation, format_table
from flight_delay.models.train import SCENARIOS, build_gradient_boosting, evaluate_all, load_split

EXPERIMENT = "flight-delay-classification"
ANALYSIS_EXPERIMENT = "flight-delay-analysis"

#: Cost of a missed delay relative to a false alarm. Not measurable from this
#: data, so the optimum is reported across a range instead of guessed at once.
COST_RATIOS = (1.0, 2.0, 3.0, 5.0, 10.0)

Splitter = Callable[[pd.DataFrame, np.ndarray[Any, Any]], TemporalHoldout]


def _show_calibration(result: Evaluation) -> None:
    log(f"    {'bin':>12s} {'count':>10s} {'predicted':>10s} {'observed':>10s} {'gap':>8s}")
    for b in result.calibration:
        log(
            f"    [{b.lower:.1f},{b.upper:.1f}) {b.count:>10,} "
            f"{b.mean_predicted:10.4f} {b.observed_rate:10.4f} {b.gap:8.4f}"
        )


def cmd_train(paths: Paths) -> int:
    """Both scenarios, baselines first, logged to MLflow."""
    import mlflow

    start_experiment(paths, EXPERIMENT)
    summary: dict[str, dict[str, dict[str, float]]] = {}

    with duckdb_connection(paths) as con:
        for scenario, features in SCENARIOS.items():
            log(f"\n{'=' * 78}\n== scenario {scenario}  ({len(features)} features)\n{'=' * 78}")

            split = load_split(con, paths, features)
            log(
                f"train={len(split.train_x):,}  test={len(split.test_x):,}  "
                f"train base rate={split.base_rate:.4f}"
            )

            start = time.perf_counter()
            results = evaluate_all(split, features)
            log(f"fitted and scored in {time.perf_counter() - start:.0f}s\n")
            log(format_table(results))

            summary[scenario] = {}
            for result in results:
                with mlflow.start_run(run_name=f"{scenario} | {result.name}"):
                    mlflow.log_params(
                        {
                            "scenario": scenario,
                            "model": result.name,
                            "n_features": len(features),
                            "train_year": 2023,
                            "test_year": 2024,
                            "split": "temporal",
                        }
                    )
                    mlflow.log_metrics(result.as_dict())
                summary[scenario][result.name] = result.as_dict()

            best = max(results, key=lambda r: r.pr_auc)
            log(
                f"\n  best: {best.name}  PR-AUC {best.pr_auc:.4f} "
                f"(lift {best.lift_over_base_rate:.2f}x)"
            )
            log("\n  calibration of the best model:")
            _show_calibration(best)

    write_result(paths, "classification.json", summary)
    return 0


def cmd_calibrate(paths: Paths) -> int:
    """Compare two ways of reserving a calibration set. Both made it worse."""
    import mlflow

    start_experiment(paths, ANALYSIS_EXPERIMENT)
    features = SCENARIO_A_FEATURES
    # Both splitters take (frame, labels) and return a TemporalHoldout; the
    # annotation is what lets mypy see that through the dict.
    strategies: dict[str, Splitter] = {
        "last two months (Nov-Dec)": split_by_month,
        "15% of every month": split_stratified_by_month,
    }
    results: list[Evaluation] = []
    summary: dict[str, dict[str, float]] = {}

    with duckdb_connection(paths) as con:
        split = load_split(con, paths, features)

    for name, splitter in strategies.items():
        holdout = splitter(split.train_x, split.train_y)
        n_fit, n_calib = holdout.sizes
        log(f"\n== {name}:  fit={n_fit:,}  calibrate={n_calib:,} ==")

        model = build_gradient_boosting(features)
        model.fit(holdout.fit_x, holdout.fit_y)

        raw = model.predict_proba(split.test_x)[:, 1]
        before = _evaluate(f"raw ({name})", split.test_y, raw)

        calibrator = IsotonicCalibrator(model).fit(holdout.calib_x, holdout.calib_y)
        mapped = calibrator.predict_proba_positive(split.test_x)
        after = _evaluate(f"isotonic ({name})", split.test_y, mapped)

        log(f"  ranking preserved: {ranking_is_preserved(raw, mapped)}")
        log(f"  Brier     {before.brier:.4f} -> {after.brier:.4f}")
        log(f"  worst gap {before.max_calibration_gap:.4f} -> {after.max_calibration_gap:.4f}")
        _show_calibration(after)

        results.extend([before, after])
        summary[name] = {
            "brier_before": before.brier,
            "brier_after": after.brier,
            "gap_before": before.max_calibration_gap,
            "gap_after": after.max_calibration_gap,
            "pr_auc_before": before.pr_auc,
            "pr_auc_after": after.pr_auc,
        }
        with mlflow.start_run(run_name=f"A | isotonic, {name}"):
            mlflow.log_params(
                {
                    "scenario": "A_pre_departure",
                    "calibration_holdout": name,
                    "fit_rows": n_fit,
                    "calibration_rows": n_calib,
                }
            )
            mlflow.log_metrics(after.as_dict())
            mlflow.log_metrics({f"raw_{k}": v for k, v in before.as_dict().items()})

    log("\n\n== all four, side by side ==")
    log(format_table(results))
    write_result(paths, "calibration.json", summary)
    return 0


def _evaluate(name: str, y_true: Any, y_prob: Any) -> Evaluation:
    from flight_delay.models.metrics import evaluate

    return evaluate(name, y_true, y_prob)


def cmd_analyse(paths: Paths) -> int:
    """What the model uses, and where to put the threshold."""
    import mlflow

    start_experiment(paths, ANALYSIS_EXPERIMENT)
    features = SCENARIO_A_FEATURES

    with duckdb_connection(paths) as con:
        split = load_split(con, paths, features)

    holdout = split_stratified_by_month(split.train_x, split.train_y)
    model = build_gradient_boosting(features)
    model.fit(holdout.fit_x, holdout.fit_y)
    probabilities = model.predict_proba(split.test_x)[:, 1]
    result = _evaluate("gradient boosting", split.test_y, probabilities)
    log(f"PR-AUC {result.pr_auc:.4f}  Brier {result.brier:.4f}")

    log("\n== permutation importance (PR-AUC drop, 200k sampled test rows) ==")
    start = time.perf_counter()
    importances = permutation_importance(
        lambda frame: model.predict_proba(frame)[:, 1],
        split.test_x,
        split.test_y,
        features=features,
        repeats=3,
    )
    log(f"  computed in {time.perf_counter() - start:.0f}s\n")
    log(format_importances(importances, result.pr_auc))

    dead = [i.feature for i in importances if abs(i.mean_drop) < 1e-5]
    if dead:
        log(f"\n  contributing nothing measurable: {', '.join(dead)}")

    log("\n== operating points ==")
    points = sweep(split.test_y, probabilities, np.round(np.arange(0.10, 0.75, 0.05), 2))
    log(
        f"  {'thr':>5s} {'alerts':>11s} {'alert rate':>11s} {'precision':>10s} "
        f"{'recall':>8s} {'F1':>7s}"
    )
    for p in points:
        log(
            f"  {p.threshold:5.2f} {p.predicted_positive:>11,} {p.alert_rate:10.1%} "
            f"{p.precision:10.3f} {p.recall:8.3f} {p.f1:7.3f}"
        )

    log("\n== optimum per assumed cost of a missed delay ==")
    best = cost_sensitivity(points, COST_RATIOS)
    log(
        f"  {'FN cost':>8s} {'threshold':>10s} {'alert rate':>11s} {'precision':>10s} "
        f"{'recall':>8s}"
    )
    for ratio, point in best.items():
        log(
            f"  {ratio:8.0f} {point.threshold:10.2f} {point.alert_rate:10.1%} "
            f"{point.precision:10.3f} {point.recall:8.3f}"
        )

    summary = {
        "metrics": result.as_dict(),
        "importances": [
            {"feature": i.feature, "pr_auc_drop": round(i.mean_drop, 6)} for i in importances
        ],
        "operating_points": [
            {
                "threshold": p.threshold,
                "precision": round(p.precision, 4),
                "recall": round(p.recall, 4),
                "alert_rate": round(p.alert_rate, 4),
            }
            for p in points
        ],
        "cost_optimum": {
            str(r): {
                "threshold": p.threshold,
                "precision": round(p.precision, 4),
                "recall": round(p.recall, 4),
            }
            for r, p in best.items()
        },
    }
    with mlflow.start_run(run_name="A | importance and thresholds"):
        mlflow.log_metrics(result.as_dict())
        mlflow.log_dict(summary, "analysis.json")
    write_result(paths, "analysis.json", summary)
    return 0
