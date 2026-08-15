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

## What this does not show

- One machine, local mode. Nothing here says anything about shuffle behaviour
  across a real cluster.
- 445 MB is small. Partition pruning matters at any size; clustering matters
  most when a partition holds many files, which needs either more data or
  smaller target files than `OPTIMIZE` will produce here.
- Reading fewer bytes is not automatically proportional wall-clock: at this
  scale the data fits in page cache, which is exactly why bytes are reported.
