# Layout benchmarks

Every number comes from Spark's own scan metrics — `numFiles` and `filesSize`
on the executed plan — not from wall-clock and not from an estimate.

That distinction is the whole point. In the first run a full table scan took
2.10 s before `OPTIMIZE` and 0.48 s after, while reading **exactly the same
bytes**. The speedup was the OS page cache, not the layout. Wall-clock alone
would have reported a 4x win that does not exist.

Machine: WSL2 on a laptop, 8 threads, 7.6 GB RAM. Spark 4.1.1, Delta 4.3.1,
local mode. Table: 13,926,960 flights partitioned by `Year`/`Month`.

## Partition pruning

`Year` and `Month` are partition columns, so a filter on them skips whole
directories without opening a file.

| Query | Files read | Bytes read | Bytes skipped |
|---|---|---|---|
| full table scan | 24 | 449.5 MB | — |
| `Year = 2024` | 12 | 229.9 MB | 48.9% |
| `Year = 2024 AND Month = 7` | **1** | **21.0 MB** | **95.3%** |

This is the cheap, reliable win, and it is available because the partition
columns were chosen to match how the data arrives and how it is queried.

## Clustering, and what Z-ordering actually buys

`Origin` cannot be a partition column: ~380 airports times 24 months would
shatter the table into tiny files. A filter on it reads everything.

**`OPTIMIZE ... ZORDER BY` could not be isolated at this table size.** Each
month is roughly 19 MB, and `OPTIMIZE` compacts a partition into a single file
whatever the file-size knobs say — it turned 291 files into 24. A partition
holding one file has nothing to skip, because that file's `Origin` range spans
the whole alphabet. Measured: 0.0% skipped, before and after.

Two knobs were tried and neither helps on open-source Delta:
`delta.targetFileSize` is a Databricks-only table property that Delta 4.3.1
rejects with `DELTA_UNKNOWN_CONFIGURATION`, and the session config
`spark.databricks.delta.optimize.maxFileSize` did not change the output.

So the mechanism was measured directly instead, with a cleaner experiment:
**two Delta tables identical in every respect except row order within each
partition.** Same partitioning, same `maxRecordsPerFile`, same data, 291 files
and ~444 MB each. One variable.

| Query | Natural order | Clustered by `Origin` | Bytes skipped |
|---|---|---|---|
| `Origin = 'ATL'` | 291 files, 444.7 MB | **50 files, 79.5 MB** | **82.1%** |
| `Origin = 'JFK'` | 291 files, 444.7 MB | 191 files, 304.7 MB | 31.5% |
| `Origin IN ('ATL','SEA')` | 291 files, 444.7 MB | 289 files, 442.8 MB | **0.4%** |
| `Origin = 'JFK' AND Year = 2024` | 147 files, 228.6 MB | 101 files, 160.8 MB | 29.7% |

Every query returned the identical answer from both tables.

Three things worth reading off that table:

**Clustering works, and it is a layout change, not an index.** Delta records
per-file min/max statistics in its transaction log. Sorting narrows each file's
`Origin` range, so the reader can prove a file cannot contain `'ATL'` and never
opens it. 82% of the bytes go untouched.

**The benefit is uneven.** `ATL` skips 82%, `JFK` only 31.5%. Sorting puts
high-volume early-alphabet values in few contiguous files; a mid-alphabet
airport is spread across more of them.

**A single sort key collapses on multi-value predicates.** `Origin IN
('ATL','SEA')` skips 0.4% — essentially nothing — because the two values sit at
opposite ends of the sort order, so nearly every file's `[min, max]` range
spans both and none can be excluded.

That last row is the argument for Z-ordering rather than sorting. A
single-column sort gives one dimension of locality; a Z-order curve interleaves
several columns so that locality survives in more than one direction at once.
Sorting is the special case that happens to work when the predicate is a single
point in a single column.

## DuckDB against Spark

Identical SQL strings, executed verbatim on both engines over the same curated
Parquet, at growing slices of the feed. Nothing is rewritten per engine. Median
of three runs; Spark's session startup is reported separately rather than
averaged into query times.

**Workload A — grouped aggregation** (daily delay rate per airport, the series
the time-series model consumes):

| Months | Rows in | DuckDB | Spark | Spark / DuckDB |
|---|---|---|---|---|
| 1 | 0.5M | 0.06 s | 0.89 s | 15.2x |
| 3 | 1.7M | 0.12 s | 0.88 s | 7.3x |
| 6 | 3.4M | 0.23 s | 1.44 s | 6.3x |
| 12 | 6.8M | 0.46 s | 2.31 s | 5.0x |
| 24 | 13.9M | **0.81 s** | 3.83 s | 4.8x |

**Workload B — window function** (`lag` over aircraft rotation, partitioned by
`Tail_Number`; a wide transformation that shuffles):

| Months | DuckDB | Spark | Spark / DuckDB |
|---|---|---|---|
| 1 | 0.11 s | 1.54 s | 13.8x |
| 3 | 0.33 s | 2.71 s | 8.3x |
| 6 | 0.92 s | 3.80 s | 4.1x |
| 12 | 2.51 s | 7.54 s | 3.0x |
| 24 | **7.18 s** | 18.31 s | **2.5x** |

Startup: DuckDB 0.03 s, Spark **17.35 s**. For a one-shot job at these sizes,
Spark spends more time starting than DuckDB spends finishing.

Both engines returned identical answers at every scale.

### Where is the crossover?

**There isn't one inside the measured range.** DuckDB is faster at every scale
on both workloads.

What does change is the gap. On the shuffle-heavy workload it narrows from
13.8x to 2.5x across 24x more data: DuckDB grew 65x while Spark grew 12x. From
12 to 24 months DuckDB took 2.86x longer for 2x the data — superlinear, it is
sort-bound — against Spark's 2.43x. The lines are converging.

Extrapolating that to a crossing point would be guessing from five points, so
the honest conclusion is a different one: **at this scale the choice is not
about speed.** DuckDB wins on speed and will keep winning until the working set
stops fitting on one machine. What Spark offers here is not a faster answer but
a different guarantee — the work is not bounded by this laptop's 7.6 GB, and it
comes with Delta, a catalog and a cluster. The crossover is about memory and
operational shape, not row count.

Which is why this project ships both, and why 13.9M rows are described as
13.9M rows rather than as "big data".

### How far away is "ever"?

`flight-delay bench-scale`

"The crossover is about memory" is only useful with a number attached, and the
number needs the size of the feed the project did *not* take. That is
measurable without downloading it: **one HTTP `HEAD` per monthly archive, read
`Content-Length`.**

| | |
|---|---|
| Archives probed | 456 (Jan 1987 – Dec 2024) |
| Archives published | **327**, October 1987 to December 2024 |
| Total, compressed | **7.70 GB** |
| Absent under this name | 1990–1999 |
| Downloaded for this project | 694 MB — **9% of the feed** |

Calibrating on the 24 months where both numbers are known exactly:

| Measured | Value |
|---|---|
| Rows per compressed GB | 20.1M |
| Curated Parquet per row | 22.88 bytes |
| Rows at each benchmark scale | 538,837 / 1,621,908 / 3,340,569 / 6,847,899 / 13,926,960 |

| Projected (estimate) | Value |
|---|---|
| Whole feed | **~155M rows**, 11.1x this project |
| Curated | **~3.5 GB** of Parquet |
| Working set, window workload (Parquet x3) | ~10.6 GB |
| This machine | 8.2 GB — **does not fit** |
| One machine runs out at | **~120M rows** |

**So a cluster would pay off somewhere before the feed's full history, and for
memory rather than for speed.** Two assumptions are load-bearing and stated
rather than buried: that 1987–2022 compresses like 2023–24, and that the
working set is about three times the Parquet on disk. The 1990s being absent
from this URL pattern means the real feed is, if anything, larger.

Note what this is *not*: it is not the ratio curves extended to where they
cross. This is a separate measurement of a different bound.

### What the trend alone would say

The dashboard draws the last measured slope forward as a dashed line, because
"the lines converge" is easier to argue about with a number attached. Taking
the final two scales in log-log space:

| Workload | Projected parity |
|---|---|
| Grouped aggregation | ~89M rows |
| Window function | ~448M rows |

Both sit past everything measured, and the memory bound (~120M rows) falls
between them. **The projection is fragile and is labelled as such** — rounding
the timings to two decimals moves the window-function crossing by roughly 25M
rows, which is why it is drawn dashed and never quoted as a result.

### A bug this comparison caught

Running both engines on the same SQL and comparing checksums found something a
single engine never would: at 3 and 6 months the two disagreed, by 2 and 1 rows
out of millions.

The cause was the window's `ORDER BY`. 2,408 groups in the feed share a
`Tail_Number`, `FlightDate` and `CRSDepTime` — 4,819 rows, physically
impossible for one airframe and so a reporting artifact. With ties, `lag` has
no defined answer, and each engine picked a different row as "previous". Where
that row's `ArrDelay` was null and the other's was not, the aggregate moved.

This is not a benchmark problem. `lag` over aircraft rotation is a **model
feature**, so the same non-determinism would have made training data depend on
which engine built it. Adding `Flight_Number`, `Origin` and `Dest` to the
ordering makes it total — zero remaining ties — and the two engines now agree
exactly at every scale.

### A portability trap

The same SQL is not portable by default. **Spark reads `"ArrDel15"` as the
string literal, not as the identifier**, so `avg("ArrDel15")` averages a
constant. Here it raised `CAST_INVALID_INPUT`, but a numeric-looking column
name would have quietly returned a wrong number. The session now sets
`spark.sql.ansi.doubleQuotedIdentifiers=true`.

## What this does not show

- One machine, local mode. Nothing here says anything about shuffle behaviour
  across a real cluster.
- 445 MB is small. Partition pruning matters at any size; clustering matters
  most when a partition holds many files, which needs either more data or
  smaller target files than `OPTIMIZE` will produce here.
- Reading fewer bytes is not automatically proportional wall-clock: at this
  scale the data fits in page cache, which is exactly why bytes are reported.
- **The engine timings move between runs**, by 10-20% on a laptop that is also
  running a browser. Re-running the whole comparison produced 0.97 s and 3.02 s
  for the 24-month aggregation against the 0.81 s and 3.83 s quoted above, and
  4.87 s against 7.18 s for the window function. Every ratio kept its direction
  and every answer still matched across engines, so the conclusion is
  unaffected — but a single decimal here is not a measurement, and
  `artifacts/results/engines.json` holds whichever run wrote it last.
