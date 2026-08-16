"""Export the pre-departure model as a deployable bundle.

The model is fitted on 2023 only, the same fit whose 2024 score is quoted, so
the number attached to the artifact is one it actually earned. Refitting on
both years would very likely serve better and would leave nothing honest to
report, so that trade is made explicitly rather than silently.
"""

from __future__ import annotations

import time
from typing import Any

from flight_delay.commands._common import duckdb_connection, log, start_experiment
from flight_delay.config import Paths
from flight_delay.features.build import SCENARIO_A_FEATURES
from flight_delay.models.metrics import evaluate
from flight_delay.models.train import TRAIN_YEAR, build_gradient_boosting, load_split
from flight_delay.serving import bundle

EXPERIMENT = "flight-delay-serving"
REGISTERED_NAME = "flight-delay-pre-departure"

#: Types skops must be told are safe to reconstruct. Reviewed one by one:
#: the project's own category grouper, and two sklearn internals it closes
#: over. Anything appearing here that is not on this list should stop a
#: release, not be waved through.
TRUSTED_TYPES = [
    "flight_delay.models.train.RareCategoryGrouper",
    "functools.partial",
    "sklearn.utils.validation.check_array",
]


def _prior_rates(con: Any, paths: Paths) -> Any:
    """Most recent monthly rate per origin and per carrier, for serving.

    Training uses the *previous* month's rate for each row. At request time the
    most recent month available plays that role, which is the closest thing to
    the training definition that exists when the flight has not happened yet.
    """
    src = f"read_parquet('{paths.table('model_table')}/**/*.parquet', hive_partitioning=true)"
    # `window` is a reserved word in DuckDB, hence `recent`.
    return con.execute(
        f"""
        WITH latest AS (SELECT max(flight_date) AS d FROM {src}),
        recent AS (
            SELECT * FROM {src}
            WHERE flight_date > (SELECT d FROM latest) - INTERVAL 30 DAY
        )
        SELECT 'origin' AS kind, origin AS key,
               avg(label) AS rate, count(*) AS flights
        FROM recent GROUP BY 1, 2
        UNION ALL
        SELECT 'carrier' AS kind, carrier AS key,
               avg(label) AS rate, count(*) AS flights
        FROM recent GROUP BY 1, 2
        """
    ).df()


def cmd_export_model(paths: Paths) -> int:
    import mlflow

    start_experiment(paths, EXPERIMENT)
    features = SCENARIO_A_FEATURES

    with duckdb_connection(paths) as con:
        split = load_split(con, paths, features)
        priors = _prior_rates(con, paths)

    log(f"fitting on {TRAIN_YEAR} ({len(split.train_x):,} flights)")
    start = time.perf_counter()
    model = build_gradient_boosting(features)
    model.fit(split.train_x, split.train_y)
    log(f"  fitted in {time.perf_counter() - start:.0f}s")

    probabilities = model.predict_proba(split.test_x)[:, 1]
    result = evaluate("gradient boosting", split.test_y, probabilities)
    log(f"  held-out 2024: PR-AUC {result.pr_auc:.4f}  Brier {result.brier:.4f}")

    metadata = {
        "scenario": "A_pre_departure",
        "train_year": TRAIN_YEAR,
        "test_year": 2024,
        "features": list(features),
        **{k: round(v, 6) for k, v in result.as_dict().items()},
    }

    target = paths.root / "model"
    bundle.save(target, model=model, priors=priors, metadata=metadata)
    log(f"  wrote bundle to {target}")

    with mlflow.start_run(run_name="pre-departure bundle"):
        mlflow.log_params(
            {"scenario": "A_pre_departure", "train_year": TRAIN_YEAR, "n_features": len(features)}
        )
        mlflow.log_metrics(result.as_dict())
        # A signature and an input example turn "some pickle" into something a
        # caller can validate against before sending a request.
        #
        # MLflow 3.15 serialises sklearn through skops, which refuses to write
        # types it does not recognise rather than trusting whatever the pickle
        # contains. The project's own transformer has to be declared, which is
        # the right default: an unreviewed list here is how a model artifact
        # becomes an execution vector.
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=split.test_x.head(3),
            registered_model_name=REGISTERED_NAME,
            skops_trusted_types=TRUSTED_TYPES,
        )
        mlflow.log_dict(metadata, "metadata.json")

    log(f"  registered as '{REGISTERED_NAME}' in MLflow")
    log("\nserve it with:\n  uvicorn flight_delay.serving.api:app --reload")
    return 0
