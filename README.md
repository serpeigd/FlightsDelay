# Flight Delay Prediction at Scale

**[Live dashboard →](https://flightsdelay-demo.streamlit.app/)**

Predicting whether a US domestic flight arrives 15+ minutes late, over
**13,926,960 flights** (BTS On-Time Performance, 2023-2024), with an explicit
**prediction cutoff** so the model never sees information that would not exist
at inference time.

Every figure below is produced by a command in this repo. Nothing is estimated.

## The result

| Scenario | What it may know | PR-AUC | Lift |
|---|---|---|---|
| **A — pre-departure** (T-2h) | schedule, route, carrier, congestion, inbound aircraft *if it has landed* | **0.343** | 1.65x |
| **B — post-departure** | everything above, plus actual departure delay | **0.938** | 4.50x |

Same label, same data, same models, same temporal split. The only difference is
whether the model may see how late the aircraft actually pushed back.

**That gap is the project.** The source feed has 109 columns and most of the
obviously predictive ones — `DepDelay`, `TaxiOut`, and the
`CarrierDelay`/`WeatherDelay`/`NASDelay`/`LateAircraftDelay` breakdown — are
recorded *after* the outcome. `LateAircraftDelay` is the clearest trap: it is an
after-the-fact accounting of the very delay being predicted.

A model built without an explicit cutoff drifts into scenario B by accident,
reports 0.94, and is useless for the question a passenger actually asks — *should
I leave for the airport?* — because at that moment nobody knows the departure
delay yet.

So [`ingest/schema.py`](src/flight_delay/ingest/schema.py) records, for every
column, **when its value becomes known** (`SCHEDULED` / `DEPARTED` / `ARRIVED` /
`LABEL`). Feature sets are assembled by filtering on that, never by listing
columns by hand, which makes leakage a contract violation rather than a
suspiciously good score.

## Fourteen things the data said

Collected in **[docs/findings.md](docs/findings.md)** — every one contradicted an
expectation and came from reading output, not from a test passing. A few:

- **Bad news about the inbound aircraft is still good news overall.** Telling
  the model the inbound is 90 minutes late *lowers* the prediction, from 31.5%
  to 22.9%, because knowing it at all means the turnaround exceeds two hours.
  Reads like a bug; is not one.

- **The clock beats the feature engineering.** Scheduled departure time accounts
  for 14.7% of PR-AUC; the painstakingly built inbound-aircraft features manage
  3.5%. Four of seventeen features measure exactly zero.
- **Textbook calibration made the model worse, twice.** Isotonic regression
  degraded Brier and tripled the worst calibration gap. Gradient boosting
  optimises a proper scoring rule, so it was already calibrated.
- **Seconds lie.** A scan took 2.10 s before `OPTIMIZE` and 0.48 s after, reading
  *exactly the same bytes*. That is the page cache, not the layout. Every
  performance claim here is measured in files and bytes read.
- **Running two engines found a bug neither would have.** DuckDB and Spark
  disagreed by 2 rows out of millions, because 4,819 rows tie on a window's
  `ORDER BY` and `lag()` is undefined under ties — in a column used as a *model
  feature*.
- **Weekly seasonality is weaker than assumed**, so the seasonal-naive forecast
  loses to simply repeating yesterday.

## Scale, honestly

13.9M rows do not need Spark, so the project measures rather than asserts:

| Workload (24 months) | DuckDB | Spark |
|---|---|---|
| Grouped aggregation | **0.81 s** | 3.83 s |
| Window function | **7.18 s** | 18.31 s |
| Startup | 0.03 s | 17.35 s |

Identical SQL on both. DuckDB wins at every scale tested; the gap narrows from
13.8x to 2.5x as the data grows 24x but never crosses. The conclusion is that
the choice is about memory and operational shape, not speed — see
[docs/benchmarks.md](docs/benchmarks.md), which also covers partition pruning
(95.3% of bytes skipped) and clustering (82.1%).

## Running it

Everything runs inside WSL. Java 17 and Python 3.12 install into `$HOME`, so no
`sudo` and no password.

```bash
scripts/setup_wsl.sh && scripts/download_bts.sh
```

```bash
source scripts/env.sh && uv sync --all-extras --group dev
```

Then the pipeline, in order. Each step prints what it produced:

```bash
flight-delay status && flight-delay extract && flight-delay curate && flight-delay features
```

```bash
flight-delay train && flight-delay analyse && flight-delay calibrate && flight-delay forecast
```

The two benchmark commands need Java and a Spark session:

```bash
flight-delay bench-layout && flight-delay bench-engines
```

Then export the deployable bundle and serve it:

```bash
flight-delay export-model
```

```bash
uvicorn flight_delay.serving.api:app
```

```bash
streamlit run src/flight_delay/serving/dashboard.py
```

### Deploying the dashboard

The data lake is 1.1 GB and lives in WSL; none of it is in git. But the
dashboard only reads a handful of JSON summaries and a 1 MB model bundle, and
those **are** committed under `artifacts/`, written by

```bash
flight-delay publish-artifacts
```

so the app runs anywhere with no lake at all. It prefers the live lake when one
exists and falls back to the committed copy otherwise, and says in the sidebar
which it used — silently preferring the committed copy would let someone stare
at stale numbers believing they had just regenerated them.

Deployed at **https://flightsdelay-demo.streamlit.app/** on
[Streamlit Community Cloud](https://share.streamlit.io): main file
`src/flight_delay/serving/dashboard.py`, installed from `requirements.txt`.
That path pulls neither MLflow nor Spark — `mlflow` sits in the `tracking`
extra precisely so the deployed app stays small.

Checks — ruff, `mypy --strict`, 144 tests:

```bash
scripts/check.sh
```

## Serving

`POST /predict` takes a scheduled flight and returns a probability, a decision
at the caller's threshold, and the base rate for context — because "38%" means
nothing to a passenger without "the average flight is 21%".

Two design choices worth naming:

**The threshold is a request parameter, not a constant.** It depends on what a
missed delay costs against a false alarm, which is a business input this project
cannot measure. At the conventional 0.5 the service warns 0.9% of passengers and
catches 2.4% of delays — close to switching itself off.

**Only the pre-departure model is served.** Scenario B scores far better and
answers a question nobody needs answered at request time.

Serving-side feature construction lives in
[`serving/features.py`](src/flight_delay/serving/features.py) and is tested
against the training SQL, since that boundary is where training-serving skew is
born. A test asserts the served columns match the training schema exactly and in
order.

## Documentation

| Document | What is in it |
|---|---|
| [findings.md](docs/findings.md) | The fourteen results worth knowing |
| [data-profile.md](docs/data-profile.md) | Volume, schema, label, data quality |
| [modelling.md](docs/modelling.md) | Both scenarios, calibration, importance, thresholds |
| [timeseries.md](docs/timeseries.md) | Daily delay rate, rolling-origin backtest |
| [benchmarks.md](docs/benchmarks.md) | Partition pruning, clustering, DuckDB vs Spark |
| [conclusions.md](docs/conclusions.md) | What the five questions answered, and what this is not |

## Layout

| Path | What |
|---|---|
| `src/flight_delay/ingest/` | Schema contract and extraction |
| `src/flight_delay/features/` | Model table, with point-in-time correctness in SQL |
| `src/flight_delay/models/` | Baselines, models, metrics, calibration, decisions |
| `src/flight_delay/timeseries/` | Daily series and rolling-origin backtest |
| `src/flight_delay/bench/` | Scan metrics, layout and engine comparisons |
| `src/flight_delay/commands/` | One module per pipeline step |

Source lives on Windows; **all data lives inside WSL** (`~/data/flight-delay`).
`scripts/bench_fs.sh` measured why: writing 300 small files takes 19 ms on the
WSL filesystem against 851-1470 ms through `/mnt/c`. OneDrive is not the
bottleneck — the Windows/Linux bridge is, by 50-100x.

## What this would look like on a cluster

Everything here runs on one laptop, which is the honest scale for 13.9M rows.
The shape it would take on managed infrastructure is worth stating, since the
techniques are the same ones and only the surfaces move.

| Here | On Databricks |
|---|---|
| Delta tables under `~/data/flight-delay/lake` | Delta tables on object storage — ADLS or S3 — behind Unity Catalog |
| `flight-delay curate` in a shell | The same command as a Job task on a job cluster, one task per pipeline step |
| SQLite MLflow backend | The workspace's managed MLflow, same API, same `log_model` call |
| `spark.driver.memory = 4g`, `local[*]` | Cluster sizing; `spark.sql.shuffle.partitions` becomes a real decision rather than a laptop compromise |
| Partition pruning measured in files read | The same measurement, and it becomes a **billing** measurement: scanned bytes are what a warehouse charges for |

Two things would change substantively rather than cosmetically:

**The engine comparison would flip eventually.** DuckDB wins here because the
working set fits in memory. At the point where it does not, the comparison in
[docs/benchmarks.md](docs/benchmarks.md) stops being about speed and starts
being about whether the job completes at all.

**Clustering would start paying.** Z-ordering did nothing at this size because
`OPTIMIZE` compacts each month into one file and skipping decides per file. With
partitions holding hundreds of files, the 82% figure measured by hand here is
what `OPTIMIZE ... ZORDER BY` would deliver on its own.

My own Databricks experience is on **Azure**, not AWS. Spark and Delta are
identical across the two; what differs is the object store and the identity
model.

## Limitations

- **No weather**, the largest external driver of delay. Everything here is a
  floor established without it.
- **PR-AUC 0.34 is not a deployable warning system.** It is an honest floor for
  the pre-departure question, not a product.
- **US domestic only**, two years. Nothing transfers to European or long-haul
  operations without refitting.
- **No hyperparameter search.** Tuning would move the numbers somewhat; it would
  not move the gap between the two scenarios, which is the finding.

## Related

[TravelPlanner](https://github.com/serpeigd/TravelPlanner) — contract-driven
travel recommender: explainable ranking, hard constraints enforced in code
before the LLM writes anything, ranking evaluation without relevance labels.

## License

MIT
