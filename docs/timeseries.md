# Forecasting the daily delay rate

A different question from the classification model. That one asks whether *this
flight* will be late; this one asks how bad *tomorrow* will be at an airport,
which is the question an operations or staffing decision actually turns on.

731 daily observations, 2023-01-01 to 2024-12-31. Fit on 2023, forecast 2024 on
a rolling origin: the model is refitted at the start of each test month on
everything strictly before it, then forecasts every day inside it. Twelve
refits, 366 national forecasts. Random cross-validation is not used — it would
train on Wednesday to predict the Tuesday before it.

**MASE is the metric**: mean absolute error divided by the in-sample error of
the seasonal naive forecast. Below 1 beats last week's value; above 1 does not.
A raw MAE of 0.05 on a rate that ranges from 0.047 to 0.617 means nothing on
its own.

## What the series looks like

| | |
|---|---|
| Days | 731 |
| Delay rate, median | 0.194 |
| Delay rate, range | 0.047 to **0.617** |

**Weekly seasonality is weaker than expected.** Mean delay rate by weekday in
2023:

| Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|
| 0.204 | **0.175** | 0.194 | 0.214 | 0.225 | 0.199 | **0.226** |

Five percentage points between the best day and the worst. That matters for
what follows: at national level the seasonal naive is a *weak* baseline, and
simply carrying yesterday's value forward beats it.

**The worst day of 2023 was 2023-01-11, at 61.7% delayed** — more than three
times the median and 22 points above the next worst day. No calendar feature
predicts a day like that, and none should pretend to. It is a single
nationwide disruption, and its main effect on this work is to remind that the
tail of this distribution is driven by events outside the data.

The ten worst days of 2023 are not clustered around holidays either: their
distances to the nearest federal holiday are -5, 0, +12, +1, +44, +34, +10,
+20, +4 and +6 days. Holiday proximity is a feature worth having, not an
explanation.

## National series

| Horizon | Forecaster | MAE | RMSE | MASE |
|---|---|---|---|---|
| 1 day | **gradient boosting** | 0.0454 | 0.0596 | **0.798** |
| | naive (yesterday) | 0.0476 | 0.0641 | 0.836 |
| | rolling 28-day mean | 0.0565 | 0.0736 | 0.993 |
| | seasonal naive (same weekday) | 0.0617 | 0.0828 | 1.084 |
| 7 days | **gradient boosting** | 0.0553 | 0.0762 | **0.987** |
| | naive / seasonal naive | 0.0617 | 0.0828 | 1.100 |
| | rolling 28-day mean | 0.0625 | 0.0823 | 1.116 |

Two things to read off this.

**One day ahead, the model beats the baselines but not by much.** MASE 0.798
against 0.836 for simply repeating yesterday. Most of what is forecastable at
one day is persistence, and the model adds about four points on top of it.

**Seven days ahead, the model is worth almost nothing.** MASE 0.987 is a 1.3%
improvement on a naive forecast. With no weather input, a week ahead is beyond
what calendar features and lagged rates can reach, and reporting 0.987 as a
success would be dressing up a negative result.

At horizon 7 the naive forecast and the seasonal naive are *identical* — seven
days ago is the same weekday last week — and both scored 1.100 to three
decimals. That is a consistency check on the backtest, not a coincidence.

## Per airport

Ten busiest origins, 7,308 airport-days, same protocol.

| Horizon | Forecaster | MAE | MASE |
|---|---|---|---|
| 1 day | **gradient boosting** | 0.0714 | **0.724** |
| | naive (yesterday) | 0.0808 | 0.819 |
| | rolling 28-day mean | 0.0824 | 0.836 |
| | seasonal naive | 0.0988 | 1.003 |
| 7 days | **gradient boosting** | 0.0829 | **0.852** |
| | rolling 28-day mean | 0.0883 | 0.908 |
| | naive / seasonal naive | 0.0988 | 1.017 |

**The model earns its place at airport level, unlike at national level.** MASE
0.724 at one day, and 0.852 even at seven — a materially better margin than the
0.987 the national series managed.

Per airport, one day ahead, all ten beat the seasonal naive:

| Airport | MASE (h=1) | MASE (h=7) |
|---|---|---|
| LAS | 0.659 | 0.740 |
| MCO | 0.659 | 0.772 |
| DEN | 0.694 | 0.823 |
| LAX | 0.718 | 0.812 |
| ORD | 0.721 | 0.801 |
| PHX | 0.723 | 0.880 |
| DFW | 0.801 | 0.943 |
| CLT | 0.824 | 0.985 |
| ATL | 0.842 | **1.000** |
| SEA | 0.863 | 0.955 |

**At seven days, ATL is a tie and CLT and SEA are close to one.** Atlanta is
the busiest airport in the feed and the one with the *lowest* delay rate
(19.7%), which leaves less variance to explain; the model recovers none of it a
week out. Reporting the pooled 0.852 without this table would hide that the
result is uneven.

## Honest limits

- **No weather.** The single largest driver of daily delay variation is absent.
  Everything above is a floor established with calendar and history only, and
  the 7-day results in particular are close to what that floor allows.
- **Extreme days are not forecastable here.** The worst day in the data is 3.2x
  the median. A model tuned to MAE will never predict it, and MASE will not
  reward trying.
- **Twelve refits, not 366.** The origin advances monthly rather than daily.
  Refitting every day is more faithful to deployment and costs 30x more; the
  block size is stated rather than hidden, and the conclusion — a large margin
  at h=1, a marginal one at h=7 — is not close enough for the difference to
  change it.
- **MASE is scaled per series**, so the national and per-airport numbers are not
  directly comparable to each other. Airport series are noisier, which inflates
  the denominator.
