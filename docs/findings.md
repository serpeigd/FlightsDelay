# What the data actually said

Every item here contradicted an expectation, and every one came from reading
output rather than from a test passing. They are collected in one place because
they are the substance of the project: the pipeline is scaffolding, these are
the result.

Each names the command that produces it.

Figures are quoted to three decimals. The fourth is not reproducible — see the
last item.

---

## 1. Knowing the departure delay is almost the entire problem

`flight-delay train`

| Scenario | PR-AUC | Lift over base rate |
|---|---|---|
| Pre-departure (T-2h) | **0.343** | 1.65x |
| Post-departure | **0.938** | 4.50x |

Same label, same data, same models, same split. The only difference is whether
the model may see how late the aircraft actually pushed back.

A model built without an explicit cutoff slides into the second row by
accident, reports 0.94, and cannot answer the question a passenger asks —
*should I leave for the airport?* — because at that moment nobody knows the
departure delay yet. This is why column availability is encoded in the schema
rather than in a comment.

## 2. The clock beats the feature engineering

`flight-delay analyse`

Permutation importance, pre-departure scenario:

| Feature | Share of PR-AUC |
|---|---|
| Scheduled departure time | **14.7%** |
| Month | 7.3% |
| Origin airport | 5.9% |
| Carrier | 5.0% |
| Inbound aircraft delay | 3.5% |

The single most predictive thing available before departure is *what time the
flight is scheduled*. Delay accumulates through the operating day, and an
8am departure is a different proposition from a 7pm one.

The inbound-aircraft features — the most carefully built in the project, with
their own point-in-time correctness rules — come fourth and fifth. That is a
deflating result and it is reported as measured.

**Four of seventeen features contribute nothing at all.** `dep_hour` is
`dep_minute_of_day / 60`. `is_weekend` is recoverable from `day_of_week`.
`inbound_known` is implied by whether `inbound_delay` is null. Redundancy that
looked reasonable when written measured exactly zero.

## 3. Textbook calibration made the model worse — twice

`flight-delay calibrate`

| Calibration holdout | Brier | Worst calibration gap |
|---|---|---|
| Last two months | 0.1555 → 0.1563 | 0.1465 → **0.3218** |
| 15% of every month | 0.1552 → 0.1556 | 0.1003 → **0.1583** |

Isotonic regression is the standard fix for overconfident probabilities. It
was applied, measured, and it degraded both setups. Gradient boosting
optimises log loss — a proper scoring rule — so it was already calibrated;
there was no systematic bias to remove and the mapping only added variance.

**The stricter split also produced the worse model.** Reserving November and
December is chronologically the cleanest choice and it leaves the model blind
to the holiday peak, which the test year contains. Its *uncalibrated* gap is
0.1465 against 0.1003 for a model that saw a slice of every month. More rigour
in one place bought less accuracy in another.

## 4. Z-ordering did nothing until the file layout allowed it to

`flight-delay bench-layout`

Partition pruning works exactly as advertised: filtering to one month reads
**1 file and 21 MB instead of 24 files and 449 MB**, skipping 95.3% of the
bytes.

`OPTIMIZE ... ZORDER BY` did nothing at all — 0.0% skipped, before and after.
Each month is roughly 19 MB and `OPTIMIZE` compacts a partition into a single
file whatever the size settings say. Data skipping decides per *file*, so a
partition holding one file has nothing to skip: that file's `Origin` range
spans the whole alphabet regardless of how its rows are sorted.

Measured properly, with two tables identical but for row order:

| Query | Natural order | Clustered by origin | Bytes skipped |
|---|---|---|---|
| `Origin = 'ATL'` | 291 files, 444.7 MB | 50 files, 79.5 MB | **82.1%** |
| `Origin = 'JFK'` | 291 files, 444.7 MB | 191 files, 304.7 MB | 31.5% |
| `Origin IN ('ATL','SEA')` | 291 files, 444.7 MB | 289 files, 442.8 MB | **0.4%** |

The last row is the interesting one. Two values at opposite ends of a single
sort key leave nearly every file's min/max spanning both, so nothing can be
excluded. That is precisely the gap a Z-order curve exists to close, and why
sorting on one column is not a substitute for it.

## 5. Seconds lie; bytes do not

`flight-delay bench-layout`

A full table scan took **2.10 s before `OPTIMIZE` and 0.48 s after, reading
exactly the same bytes**. The speedup was the operating system's page cache.

Reported as wall-clock, that is a 4x improvement that does not exist. Every
layout claim in this project is therefore measured in files and bytes read,
taken from Spark's own scan metrics.

## 6. 13.9M rows never needed Spark

`flight-delay bench-engines`

Identical SQL, same data, both engines:

| Workload (24 months) | DuckDB | Spark | Ratio |
|---|---|---|---|
| Grouped aggregation | **0.81 s** | 3.83 s | 4.8x |
| Window function | **7.18 s** | 18.31 s | 2.5x |
| Startup | 0.03 s | **17.35 s** | — |

DuckDB wins at every scale tested. The gap narrows from 13.8x to 2.5x as the
data grows 24x, so the lines converge — but they do not cross, and
extrapolating five points to a crossing point would be guessing.

The honest conclusion is that the choice is not about speed. DuckDB keeps
winning until the working set stops fitting in one machine's memory. What
Spark buys here is a different bound, not a faster answer.

## 7. Running two engines found a bug neither would have

`flight-delay bench-engines`

Comparing checksums across engines, the answers differed at 3 and 6 months —
by 2 rows and 1 row out of millions.

The cause: 2,408 groups in the feed share a tail number, flight date and
scheduled departure time — 4,819 rows, physically impossible for one aircraft
and therefore a reporting artifact. With ties, `lag()` has no defined answer,
and DuckDB and Spark each picked a different "previous flight".

That expression is **a model feature**, not just a benchmark query. The
training data would have depended on which engine built it. Adding flight
number, origin and destination to the ordering makes it total — zero ties
remain, and the engines now agree exactly at every scale.

## 8. Weekly seasonality is weaker than the textbook assumes

`flight-delay forecast`

Mean delay rate by weekday, 2023:

| Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|
| 0.204 | **0.175** | 0.194 | 0.214 | 0.225 | 0.199 | **0.226** |

Five percentage points from best to worst. The consequence is concrete: at
national level the seasonal naive forecast (*same weekday last week*) **loses
to simply repeating yesterday's value** — MASE 1.084 against 0.836.

The standard advice to use a seasonal baseline for daily series is wrong here,
and only measuring shows it.

## 9. A week ahead is beyond reach without weather

`flight-delay forecast`

| Series | Horizon | MASE | Verdict |
|---|---|---|---|
| National | 1 day | 0.798 | useful |
| National | 7 days | **0.987** | 1.3% better than nothing |
| Airports | 1 day | 0.724 | useful |
| Airports | 7 days | 0.852 | useful, unevenly |

The 7-day national forecast is a negative result and is written up as one.
Per airport the model does earn its place — but at 7 days **Atlanta ties the
baseline exactly at MASE 1.000**, with Charlotte at 0.985 and Seattle at 0.955.
Atlanta is the busiest airport in the feed and has the *lowest* delay rate,
leaving least variance to explain.

Reporting the pooled 0.852 without that breakdown would hide it.

## 10. One flight departed and never landed, and it is in the data

`flight-delay features`

Of 13,926,960 flights, 218,310 carry no label. 184,258 are cancelled and
34,051 diverted. **Exactly one is neither.**

A flight from LaGuardia to Nashville in May 2023 that pushed back 37 minutes
late, taxied for 36 minutes, has a recorded wheels-off time — and then no
arrival record at all. `Cancelled` and `Diverted` are both zero.

It is a reporting gap, not a mystery. It is named here rather than filtered
away silently, because a pipeline that drops one row without saying so is a
pipeline that will drop a million the same way.

## 11. The feed has a phantom column

`flight-delay curate`

Every header and data line ends with a trailing comma, producing a 110th
unnamed field carrying nothing. A positional schema that ignores it shifts
every column by one, and nothing errors — the data is simply wrong.

It is named `_trailing` and dropped. The headers of all 24 archives are checked
against each other before a byte is written, rather than trusting the first.

## 12. Only 27% of flights know their inbound aircraft in time

`flight-delay features`

The strongest available pre-departure signal is how late the same airframe's
previous leg landed. At two hours before departure that aircraft is usually
still in the air: typical turnarounds are under an hour.

| Inbound status | Flights | Delayed |
|---|---|---|
| Landed on time or early | 2,277,360 | 12.3% |
| Landed 1-15 min late | 581,262 | 13.6% |
| Landed 16-60 min late | 539,219 | 14.9% |
| Landed 60+ min late | 316,758 | 21.9% |
| **Not known at cutoff** | **9,994,051** | **23.3%** |

The signal is real and monotonic. But the most common case by far is not
knowing, and those flights are the *most* delayed — which is why "the inbound
is not in yet" is kept as a feature rather than dropped as a missing value.

The longest gap between an aircraft's legs in the feed is **664 days**. That is
not an inbound aircraft, it is an airframe returning from storage; correlation
with the label peaks at a 12-24h gap and collapses past three days, so the
feature is cut off at a day.

## 13. Bad news about the inbound is still good news overall

`uvicorn flight_delay.serving.api:app`

Scoring the same evening flight from JFK four ways:

| Request | Delay probability |
|---|---|
| 06:00 departure, inbound unknown | 17.8% |
| 19:30 departure, inbound unknown | **31.5%** |
| 19:30 departure, inbound landed 90 min late | **22.9%** |
| 19:30 departure, inbound landed on time | 18.2% |

Telling the model the aircraft is arriving *an hour and a half late* makes the
prediction go **down**, from 31.5% to 22.9%. That reads like a bug and is not
one.

Knowing the inbound at all means the turnaround exceeds two hours, and those
flights are delayed 13.7% of the time against 23.3% for the rest. The aircraft
has slack. So the flag carries more signal than the number it accompanies, and
a very late inbound only partially cancels the good news of there being one.

Within the group that has a usable inbound the ordering is exactly right — on
time 18.2%, ninety minutes late 22.9%. The apparent paradox is a comparison
across two different populations, and it is the sort of thing that gets a model
pulled from production by someone who spot-checks one case and does not look at
the second.

## 14. "Deterministic" was not quite deterministic

`flight-delay train` then `flight-delay export-model`

The same estimator, the same data, `random_state=0`, early stopping off — and
PR-AUC came out **0.3431** in one run and **0.3426** in another.

The cause is threaded float accumulation while building histograms: the order
in which partial sums combine is not fixed, and the result differs in the
fourth decimal. It is well inside noise and it does mean the fourth decimal of
any figure here is not reproducible. Numbers are therefore quoted to three
decimals, which is the precision that actually exists.
