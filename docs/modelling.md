# Classification results

Fit on 2023 (6,743,403 flights), evaluated on 2024 (6,965,247). Temporal split,
never random. Base rate 20.56% in train, 20.82% in test.

PR-AUC is the headline because the label is imbalanced: a constant predictor
scores PR-AUC equal to the positive rate, so **lift** below is PR-AUC divided
by that floor. Accuracy is not reported — predicting "never late" scores 79%.

## The result the project exists to produce

| Scenario | Best PR-AUC | Lift | ROC-AUC | Brier |
|---|---|---|---|---|
| **A — pre-departure (T-2h)** | **0.3431** | 1.65x | 0.6665 | 0.1553 |
| **B — post-departure** | **0.9376** | 4.50x | 0.9689 | 0.0407 |

Same label, same data, same models, same split. The only difference is that
scenario B is allowed to know the aircraft's actual departure delay and taxi-out
time.

**Departure delay is very nearly the whole answer.** Once it is known, the
problem is close to solved. Before it exists, the problem is genuinely hard.

That gap is the reason the leakage contract is the centre of this project. A
model built without an explicit cutoff drifts into scenario B by accident,
reports PR-AUC around 0.94, and is worthless for the question a passenger
actually asks — *should I leave for the airport?* — because at that moment
nobody knows the departure delay yet.

## Full results

### Scenario A — pre-departure

Knowable at T-2h before scheduled departure: schedule, route, carrier,
previous-month congestion, and the inbound aircraft **only if it had already
landed by the cutoff**.

| Model | PR-AUC | Lift | ROC-AUC | Brier |
|---|---|---|---|---|
| baseline: train base rate | 0.2082 | 1.00 | 0.5000 | 0.1648 |
| baseline: origin prior-month rate | 0.2430 | 1.17 | 0.5576 | 0.1651 |
| baseline: carrier prior-month rate | 0.2455 | 1.18 | 0.5596 | 0.1646 |
| logistic regression | 0.3213 | 1.54 | 0.6569 | 0.1569 |
| **gradient boosting** | **0.3431** | **1.65** | 0.6665 | 0.1553 |

The baselines are not strawmen: an airport's or carrier's delay rate last month
already lifts PR-AUC by 17-18% with no model at all. Gradient boosting is worth
having — 1.65x against 1.18x — but the honest framing is that a hard problem
stayed hard.

### Scenario B — post-departure

| Model | PR-AUC | Lift | ROC-AUC | Brier |
|---|---|---|---|---|
| baseline: carrier prior-month rate | 0.2455 | 1.18 | 0.5596 | 0.1646 |
| logistic regression | 0.9340 | 4.49 | 0.9650 | 0.0416 |
| **gradient boosting** | **0.9376** | **4.50** | 0.9689 | 0.0407 |

Note how little the model choice matters here: logistic regression gets within
0.004 PR-AUC of gradient boosting. When one feature dominates, a linear model
captures it and the extra capacity buys almost nothing. In scenario A the gap
between the two is five times larger, because there the signal lives in
interactions rather than in a single column.

## Calibration

Discrimination and calibration are different properties, and both are reported.
The tables below are the observed delay rate against the predicted probability,
in equal-width bins, for the gradient boosting model.

**Scenario A** is well behaved through the bulk of its range and **overconfident
at the top**:

| Predicted bin | Flights | Mean predicted | Observed | Gap |
|---|---|---|---|---|
| [0.0, 0.1) | 1,313,370 | 0.0756 | 0.0883 | -0.013 |
| [0.1, 0.2) | 2,845,407 | 0.1468 | 0.1678 | -0.021 |
| [0.2, 0.3) | 1,608,192 | 0.2440 | 0.2532 | -0.009 |
| [0.3, 0.4) | 731,593 | 0.3431 | 0.3313 | +0.012 |
| [0.4, 0.5) | 310,301 | 0.4427 | 0.4138 | +0.029 |
| [0.5, 0.6) | 117,748 | 0.5411 | 0.4773 | +0.064 |
| [0.6, 0.7) | 31,137 | 0.6367 | 0.5330 | **+0.104** |
| [0.7, 0.8) | 4,146 | 0.7333 | 0.6274 | **+0.106** |

Where it says 64%, the truth is 53%. Those bins are small — 35,000 flights out
of 7 million — but they are exactly the flights a warning system would act on,
so the error lands where it costs most.

**Scenario B** is accurate at both extremes, where almost all its mass sits
(0.0004 gap over 4.97M flights in the lowest bin, -0.004 over 975k in the
highest), and drifts up to +0.047 through the middle.

### Recalibration was tried, and it made things worse

Isotonic regression is the textbook fix, and the expectation going in was a
better Brier score at no cost to PR-AUC. It was measured instead of assumed,
and it failed twice.

| Calibration holdout | Brier raw → isotonic | Worst gap raw → isotonic |
|---|---|---|
| Last two months (Nov-Dec 2023) | 0.1555 → 0.1563 | 0.1465 → **0.3218** |
| 15% sampled from every month | 0.1552 → **0.1556** | 0.1003 → **0.1583** |

The ranking survived in both runs, as a monotonic map must — that was checked,
not assumed. But the probabilities got worse, not better, under either way of
reserving the data.

Two things are going on, and both are worth more than a successful calibration
would have been.

**Gradient boosting was already calibrated.** It optimises log loss, which is a
proper scoring rule, so well-fitted probabilities are what it produces by
default. There was no systematic bias for isotonic regression to remove, so all
it added was variance from a step function fitted on a million points, plus
whatever the 2023-to-2024 shift does to a map learned entirely on 2023.

**Holding out whole months damaged the model itself.** Reserving November and
December is the chronologically cleanest split, and it left the model blind to
the holiday peak — a period the 2024 test set contains. Its *uncalibrated*
worst gap is 0.1465 against 0.1003 for the model that saw a slice of every
month. The stricter split produced the worse model.

So the recommendation is the opposite of the textbook one: **leave the
probabilities alone.** If calibration is revisited, the thing to fix first is
the seasonal representativeness of the holdout, not the mapping function.

## What the model is actually using

Permutation importance on scenario A: shuffle one column on held-out 2024 data
and measure the PR-AUC that disappears. Impurity-based importance was avoided
because it favours high-cardinality columns — it would rank `origin` (350
levels) above `inbound_known` (2) on cardinality alone.

| Feature | PR-AUC drop | Share of total |
|---|---|---|
| `dep_minute_of_day` | 0.0497 | 14.7% |
| `month` | 0.0247 | 7.3% |
| `origin` | 0.0200 | 5.9% |
| `carrier` | 0.0171 | 5.0% |
| `inbound_delay` | 0.0117 | 3.5% |
| `dest` | 0.0102 | 3.0% |
| `inbound_turnaround_minutes` | 0.0101 | 3.0% |
| `arr_minute_of_day` | 0.0086 | 2.6% |
| `day_of_week` | 0.0082 | 2.4% |
| `carrier_prior_rate` | 0.0029 | 0.8% |
| `crs_elapsed` | 0.0017 | 0.5% |
| `distance` | 0.0010 | 0.3% |
| `origin_prior_rate` | 0.0009 | 0.3% |
| `dep_hour`, `is_weekend`, `inbound_known` | 0.0000 | 0.0% |
| `origin_prior_flights` | -0.0001 | — |

**The time of day dominates everything else, by 2x.** Delay accumulates through
the operating day, and the scheduled departure time captures most of what a
pre-departure model can know. That is a deflating result for the feature
engineering and an honest one.

**The inbound leg matters less than its build cost suggests.** `inbound_delay`
and `inbound_turnaround_minutes` contribute 3.5% and 3.0%, well behind the
clock. They are the most carefully constructed features in the project and they
are mid-table. Note also that the three inbound columns describe one underlying
fact between them, so permutation splits their credit — the group is worth more
than any row above suggests.

**Three features contribute nothing measurable.** `dep_hour` is
`dep_minute_of_day / 60` and is pure redundancy. `is_weekend` is recoverable
from `day_of_week`. `inbound_known` is implied by whether `inbound_delay` is
null, which gradient boosting reads directly. `origin_prior_flights` scores
very slightly negative, which is noise around zero. Four of seventeen features
are dead weight and should be dropped.

## Turning a probability into a decision

PR-AUC does not say when to warn a passenger. That needs a threshold, and a
threshold needs a cost: how much worse is missing a delay than raising a false
alarm? The ratio is an input, not something this project can measure, so the
optimum is reported across a range.

| Threshold | Alerts | Alert rate | Precision | Recall |
|---|---|---|---|---|
| 0.10 | 5,792,577 | 83.2% | 0.233 | 0.932 |
| 0.15 | 3,997,878 | 57.4% | 0.272 | 0.751 |
| 0.20 | 2,560,090 | 36.8% | 0.313 | 0.552 |
| 0.25 | 1,534,422 | 22.0% | 0.355 | 0.376 |
| 0.30 | 506,865 | 7.3% | 0.435 | 0.152 |
| 0.40 | 105,862 | 1.5% | 0.515 | 0.038 |
| 0.70 | 16,802 | 0.2% | 0.639 | 0.007 |

**0.5 is not a threshold, it is a default.** At 0.5 the model warns 0.9% of
passengers and catches 2.4% of delays. With a 20.8% positive rate a calibrated
model rarely exceeds 0.5 at all, so the conventional cut-off is close to
switching the system off.

Where the optimum lands, per assumed cost of a missed delay relative to a false
alarm:

| Missed delay costs | Threshold | Alert rate | Precision | Recall |
|---|---|---|---|---|
| 1x | 0.55 | 0.9% | 0.541 | 0.023 |
| 2x | 0.30 | 7.3% | 0.435 | 0.152 |
| 3x | 0.25 | 22.0% | 0.355 | 0.376 |
| 5x | 0.15 | 57.4% | 0.272 | 0.751 |
| 10x | 0.10 | 83.2% | 0.233 | 0.932 |

**The answer moves enormously with an assumption nobody has measured.** Between
a 2x and a 5x cost ratio the alert rate goes from 7% of passengers to 57%.
Presenting a single "optimal threshold" without that ratio on the table would
be presenting an arbitrary choice as a result. This is the number to ask a
product owner for, and the honest deliverable is the curve, not a point on it.

## Choices that shape these numbers

**The inbound leg is admitted only when it had landed.** The previous flight of
the same airframe is the strongest available pre-departure signal, and it is
also the easiest place to leak: at T-2h that aircraft is often still in the air.
The value is used only when the previous leg's actual arrival timestamp is at or
before the cutoff, and a flag records whether it was known — because "the
inbound is not in yet" is itself predictive and dropping those rows would delete
the hardest cases. Only **27.2%** of flights have a usable inbound.

The measured signal is monotonic and real:

| Inbound status | Flights | Delayed |
|---|---|---|
| landed on time or early | 2,277,360 | 12.3% |
| landed 1-15 min late | 581,262 | 13.6% |
| landed 16-60 min late | 539,219 | 14.9% |
| landed 60+ min late | 316,758 | 21.9% |
| not known at cutoff | 9,994,051 | 23.3% |

**Congestion priors come from the previous month only.** An airport's delay rate
over the whole dataset would leak the future into the past. Taking the previous
calendar month is knowable at prediction time and leaves 2023-01 without a
prior — 3.85% of rows — rather than pretending otherwise.

**Early stopping is off.** sklearn's implementation holds out a *random* slice
of training rows, which would put December flights in the validation set of a
model tested on the following year. The iteration count is fixed instead.

**Rare categories are pooled from the training frame only.** Gradient boosting
refuses a categorical with more than 255 levels and the feed has 350 airports,
so the tail is pooled into `OTHER` — using 2023 frequencies, because letting
2024 traffic decide which 2023 airports the model can see is leakage by
another route.

## Limitations

- **No weather.** The single largest external driver of delay is absent. NAS and
  weather delay codes exist in the feed but are recorded after arrival, so they
  are outcome, not input. A real system would join a forecast at the cutoff.
- **PR-AUC 0.34 is not a deployable warning system on its own.** At a threshold
  useful for alerting, precision is limited. The result is a floor established
  honestly, not a product.
- **US domestic only**, two years, one carrier set. Nothing here transfers to
  European or long-haul operations without re-fitting.
- **No hyperparameter search.** Both models use defaults with a fixed iteration
  count. Tuning would move these numbers somewhat; it would not move the gap
  between scenario A and scenario B, which is the actual finding.
