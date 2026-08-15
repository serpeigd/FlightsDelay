"""Train and evaluate both scenarios, with baselines that have to be beaten.

The split is temporal and fixed: fit on 2023, evaluate on 2024. Never random.
With seasonality, propagated delay and shared weather, a random fold puts
flights from the same afternoon on both sides and the score stops meaning
anything.

Three baselines run first, and they are not strawmen. "Always predict the base
rate" fixes the floor. The origin's previous-month delay rate and the carrier's
are real predictors that need no model at all; if gradient boosting cannot beat
a monthly average, that is the finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flight_delay.config import Paths
from flight_delay.features.build import (
    CATEGORICAL,
    LABEL,
    SCENARIO_A_FEATURES,
    SCENARIO_B_FEATURES,
)
from flight_delay.models.metrics import Evaluation, evaluate

if TYPE_CHECKING:
    import duckdb

SCENARIOS: dict[str, tuple[str, ...]] = {
    "A_pre_departure": SCENARIO_A_FEATURES,
    "B_post_departure": SCENARIO_B_FEATURES,
}

TRAIN_YEAR = 2023
TEST_YEAR = 2024


@dataclass(frozen=True, slots=True)
class Split:
    train_x: pd.DataFrame
    train_y: np.ndarray[Any, Any]
    test_x: pd.DataFrame
    test_y: np.ndarray[Any, Any]

    @property
    def base_rate(self) -> float:
        return float(self.train_y.mean())


def load_split(con: duckdb.DuckDBPyConnection, paths: Paths, features: tuple[str, ...]) -> Split:
    src = f"read_parquet('{paths.table('model_table')}/**/*.parquet', hive_partitioning=true)"
    columns = ", ".join(f'"{c}"' for c in (*features, LABEL))

    def year(y: int) -> pd.DataFrame:
        return con.execute(f"SELECT {columns} FROM {src} WHERE year = {y}").df()

    train, test = year(TRAIN_YEAR), year(TEST_YEAR)
    for frame in (train, test):
        for column in CATEGORICAL:
            if column in frame.columns:
                frame[column] = frame[column].astype("category")

    return Split(
        train_x=train[list(features)],
        train_y=train[LABEL].to_numpy(dtype=np.float64),
        test_x=test[list(features)],
        test_y=test[LABEL].to_numpy(dtype=np.float64),
    )


def baseline_predictions(split: Split) -> dict[str, np.ndarray[Any, Any]]:
    """Predictions that need no fitting, only what was already known."""
    n = len(split.test_x)
    out: dict[str, np.ndarray[Any, Any]] = {
        "baseline: train base rate": np.full(n, split.base_rate),
    }
    # These columns are the previous month's observed rate, so using them
    # directly as a probability is a legitimate model-free predictor.
    for column, label in (
        ("origin_prior_rate", "baseline: origin prior-month rate"),
        ("carrier_prior_rate", "baseline: carrier prior-month rate"),
    ):
        if column in split.test_x.columns:
            values = split.test_x[column].to_numpy(dtype=np.float64)
            out[label] = np.where(np.isnan(values), split.base_rate, values)
    return out


def _numeric_columns(features: tuple[str, ...]) -> list[str]:
    return [c for c in features if c not in CATEGORICAL]


#: HistGradientBoosting refuses a categorical with more than 255 levels, and
#: the feed has 350 origins and 350 destinations. One below the limit leaves
#: room for the catch-all level.
MAX_CATEGORIES = 254
RARE = "OTHER"


class RareCategoryGrouper(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Keep the most frequent levels of each categorical, pool the rest.

    The kept levels are learned from the **training** frame only. Choosing them
    from the full dataset would let 2024 traffic decide which 2023 airports the
    model can see, which is a quiet way to leak the future into the past.
    """

    def __init__(
        self, columns: tuple[str, ...] = CATEGORICAL, max_categories: int = MAX_CATEGORIES
    ):
        self.columns = columns
        self.max_categories = max_categories

    def fit(self, X: pd.DataFrame, y: object = None) -> RareCategoryGrouper:
        self.keep_: dict[str, list[str]] = {}
        for column in self.columns:
            if column not in X.columns:
                continue
            counts = X[column].value_counts()
            self.keep_[column] = list(counts.index[: self.max_categories])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for column, keep in self.keep_.items():
            values = out[column].astype(object).where(out[column].isin(keep), RARE)
            out[column] = pd.Categorical(values, categories=[*keep, RARE])
        return out

    def coverage(self, X: pd.DataFrame) -> dict[str, float]:
        """Share of rows whose level survived, per column. Reported rather than
        assumed: pooling 96 airports is only harmless if they are small."""
        return {column: float(X[column].isin(keep).mean()) for column, keep in self.keep_.items()}


def build_logistic(features: tuple[str, ...]) -> Pipeline:
    """One-hot the categoricals, impute and scale the rest.

    ``handle_unknown='infrequent_if_exist'`` matters for a temporal split: a
    route or tail that only appears in 2024 must not crash inference.
    """
    categorical = [c for c in features if c in CATEGORICAL]
    return make_pipeline(
        ColumnTransformer(
            [
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=50,
                        sparse_output=True,
                    ),
                    categorical,
                ),
                (
                    "num",
                    make_pipeline(
                        SimpleImputer(strategy="median", add_indicator=True),
                        StandardScaler(),
                    ),
                    _numeric_columns(features),
                ),
            ],
            remainder="drop",
        ),
        LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1),
    )


def build_gradient_boosting(features: tuple[str, ...]) -> Pipeline:
    """Native categorical and NaN handling, so no encoding is needed.

    NaN is meaningful here rather than missing at random: a null
    ``inbound_delay`` means the inbound leg had not landed by the cutoff, which
    is itself predictive. Imputing it would erase that.

    Early stopping is off. sklearn's implementation holds out a **random**
    fraction of the training rows, which would put December flights in the
    validation set of a model whose test set is the following year -- the same
    temporal inconsistency the whole split exists to avoid. The iteration count
    is fixed instead, so the fit is deterministic and the only chronology in
    play is 2023 to 2024.
    """
    return make_pipeline(
        RareCategoryGrouper(),
        HistGradientBoostingClassifier(
            categorical_features=[f in CATEGORICAL for f in features],
            max_iter=200,
            learning_rate=0.1,
            early_stopping=False,
            random_state=0,
        ),
    )


def evaluate_all(split: Split, features: tuple[str, ...]) -> list[Evaluation]:
    results = [
        evaluate(name, split.test_y, prob) for name, prob in baseline_predictions(split).items()
    ]

    for name, model in (
        ("logistic regression", build_logistic(features)),
        ("gradient boosting", build_gradient_boosting(features)),
    ):
        model.fit(split.train_x, split.train_y)
        prob = model.predict_proba(split.test_x)[:, 1]
        results.append(evaluate(name, split.test_y, prob))

    return results
