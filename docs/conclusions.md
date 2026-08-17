# Conclusions

Six questions, asked in order. Two answers came back negative, one came back
mixed, and the most useful one was never about accuracy.

| # | Question | Answer | Evidence | Command |
|---|---|---|---|---|
| 1 | Can a delay be predicted before the plane moves? | Yes, weakly — and weakly is the honest ceiling here | PR-AUC **0.343**, 1.67x a base rate of 20.6% | `train` |
| 2 | How much does one leaked column change that? | It replaces the problem with an easier one | **0.938** against 0.343, same data, same model, same split | `train` |
| 3 | Do the probabilities mean what they say? | Yes where the flights are; overconfident at the top | Isotonic recalibration tried on two holdouts, **both worse** | `calibrate` |
| 4 | How bad will tomorrow be at an airport? | Answerable one day out, not one week out | MASE **0.798** national / **0.724** per airport at h=1; **0.987** at h=7 | `forecast` |
| 5 | Does 13.9M rows need a cluster? | No. One laptop beat the cluster at every size tested | DuckDB faster in **10/10** measurements; the ratio narrows to 2.5x and stops | `bench-engines` |
| 6 | Would it *ever*? | Yes — near the feed's full history, and for memory, not speed | The published feed is **7.70 GB over 327 archives ≈ 155M rows**; this machine runs out around **120M** | `bench-scale` |

## What is fair to claim

- A **two-hours-ahead** delay model that is **1.67x better than guessing**, on a
  year it never saw during training.
- Probabilities that can be **shown to a passenger**: checked against outcomes
  bin by bin, not merely scored.
- A threshold expressed in **people rather than percentages** — at 0.30, 1.19M
  passengers warned, 1.00M delays still missed.
- An engine choice **argued from measurements of this data on this machine**,
  not from a vendor benchmark.
- Every figure **reproducible by one command**, including on a machine that has
  no copy of the data (the dashboard reads committed artifacts).

## What it is not

- **No weather.** The single largest driver of delay is absent. Everything here
  is the floor that calendar features and history alone reach.
- **US domestic, 2023-24 only — 9% of the published feed.** Measured, not
  guessed: 327 monthly archives exist and total 7.70 GB compressed; this project
  took 694 MB of them.
- **Historic files, not a live feed.** No streaming ingestion, no drift
  monitoring, no retraining schedule. Designed for, not built.
- **Not deployed to users.** A demo and a FastAPI endpoint, not a service with
  an SLO or an on-call rota.
- **73% of flights never learn their inbound aircraft in time**, so the
  strongest available signal is missing exactly when it would help most.
- **The fourth decimal is not reproducible** — threaded float accumulation in
  the histogram builder moves it (finding 14).

## What another month would buy

1. **Weather at origin and destination.** The one addition likely to move PR-AUC
   rather than decorate it.
2. **SARIMAX with the same calendar regressors**, so the classical baseline
   loses for the right reason rather than for an unfair one.
3. **Drift monitoring on the served model.** The base rate moved between 2023
   and 2024 and nothing in this project would have noticed.
4. **Per-carrier evaluation.** A pooled PR-AUC can hide a model that is useless
   for one airline.
5. **A cost function from whoever owns the decision**, replacing the five
   assumed miss-to-false-alarm ratios in `flight-delay analyse`.

## Why so much of this is negative results

A model that scores 0.94 by reading the actual departure delay is the most
common way this exact problem is done wrong, and it looks excellent right up to
the moment someone asks *when* that value becomes available. The pre-departure
number, 0.343, is the smaller one and the only honest one.

The pattern repeats. Isotonic regression is the textbook fix for overconfident
probabilities, and it hurt. The seasonal naive is the textbook baseline for
daily series, and it lost to repeating yesterday. Spark is the textbook answer
to 13.9M rows, and it lost ten times out of ten.

Each of those would have shipped as an unexamined default. The ambition was
bounded on purpose — one machine, two years, public data, no weather — and
inside those bounds the aim was not the highest score. It was that every number
survives being asked where it came from.
