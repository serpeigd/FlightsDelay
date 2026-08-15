# Data profile

Every number here was produced by running the pipeline over the full feed, not
sampled or estimated. Reproduce with `flight-delay status`, `scripts/check.sh`
and the curation step.

## Volume

| | |
|---|---|
| Source | BTS On-Time Reporting Carrier On-Time Performance |
| Period | 2023-01 to 2024-12, 24 monthly archives |
| Compressed | 694 MB |
| Extracted CSV | 5.9 GB |
| **Rows** | **13,926,960** |
| Curated Parquet (ZSTD, partitioned by Year/Month) | **305 MB**, 24 files |
| Curation wall-clock (DuckDB, 8 threads) | **105.8 s** |

The curated table is **19x smaller** than the CSV it came from, which is most
of why the columnar format matters before any engine comparison is run.

## Schema

The feed carries **109 named columns plus a phantom empty field**: every header
and data line ends with a trailing comma. A positional schema that ignores it
shifts every column by one. It is named `_trailing` and dropped.

Of the 109, roughly 45 are `Div1*`-`Div5*` diverted-leg details that are almost
entirely null. **50 columns** are kept.

Clock fields arrive zero-padded (`'0030'`) and numerics as `'-3.00'`, so times
are kept as strings and converted deliberately rather than coerced to integers.

**The header is identical across all 24 archives**, verified file by file
rather than trusted from the first one.

## Type contract

All 50 columns are read as `VARCHAR` and cast explicitly with `TRY_CAST`, then
every non-empty value that became null is counted.

**Result: zero cast failures across 13,926,960 rows.** Every non-empty value in
the feed parsed into its declared type.

## Label

`ArrDel15` — arrival 15+ minutes late.

| | Rows | Share |
|---|---|---|
| Total | 13,926,960 | |
| Delayed (`ArrDel15 = 1`) | 2,836,665 | **20.37%** |
| No label | 218,310 | 1.57% |

Delay rate by year: **2023: 20.56%** (6,847,899 flights) · **2024: 20.82%**
(7,079,061 flights). Stable enough that a 2023-train / 2024-test split is not
comparing two different worlds.

### Why the label is missing

| Reason | Rows |
|---|---|
| Cancelled | 184,258 |
| Diverted | 34,051 |
| **Neither** | **1** |

Those 218,310 rows are excluded from the classification dataset, and the
exclusion is reported rather than performed silently.

The single unexplained row is worth naming. It is a flight from **LGA to BNA in
May 2023** that pushed back 37 minutes late, taxied for 36 minutes and has a
recorded wheels-off time — and then has no arrival record at all: `WheelsOn`,
`ArrTime`, `ArrDelay` and `ArrDel15` are all null, while `Cancelled` and
`Diverted` are both 0. It departed and the feed never recorded it landing. This
is a reporting gap, not a modelling case; it is dropped with the rest.

## Traffic and delay by origin

| Origin | Flights | Delayed |
|---|---|---|
| ATL | 678,373 | 19.66% |
| DFW | 598,826 | 25.62% |
| DEN | 597,630 | 24.18% |
| ORD | 538,919 | 22.89% |
| CLT | 412,655 | 25.00% |

Busiest is not latest: Atlanta handles the most flights and has the *lowest*
delay rate of the five, six points below Dallas/Fort Worth.

## Caveats

- 2023-01 is not representative of the whole feed. Profiling it alone gives
  21.7% delayed and 2.2% unlabelled, against 20.37% and 1.57% over the full
  24 months. Monthly volume ranges from 538,837 (2023-01) to 619,026 (2024-08).
- `ArrDel15` is derived by BTS from `ArrDelay`, so it inherits any reporting
  quirk in the carrier's own arrival timestamps.
