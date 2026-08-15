# Flight Delay Prediction at Scale

Predicting whether a US domestic flight arrives 15+ minutes late, over
**13.3M flights** (BTS On-Time Performance, 2023-2024), with an explicit
**prediction cutoff** so the model never sees information that would not exist
at inference time.

> Work in progress. Numbers in this README are only written down once they have
> been produced by a command in this repo.

## Why the cutoff matters

The source data has 110 columns, and most of the obviously predictive ones are
recorded *after* the outcome: `DepDelay`, `TaxiOut`, `WheelsOff`, `ArrTime`,
and the `CarrierDelay`/`WeatherDelay`/`NASDelay`/`LateAircraftDelay` breakdown,
which is an after-the-fact decomposition of the very delay being predicted.

A model given `DepDelay` scores extremely well and is useless: if you already
know the aircraft pushed back 40 minutes late, you do not need a model. So two
scenarios are built and reported side by side:

| Scenario | Known at prediction time | Business question |
|---|---|---|
| **A — pre-departure** (T-2h vs scheduled departure) | carrier, route, scheduled time block, day, distance, historical congestion, inbound aircraft rotation | Should the passenger be warned before leaving home? |
| **B — post-departure** | everything in A, plus actual departure delay and taxi-out | Given it is airborne and late, how late does it arrive? |

The gap between the two is the finding, not an inconvenience.

## Layout

| Path | What |
|---|---|
| `src/flight_delay/` | Library code |
| `tests/` | Unit tests; `-m spark` and `-m data` are opt-in |
| `scripts/` | Environment setup, download, benchmarks, checks |
| `docs/` | Decisions and measured results |

Source lives on Windows; **all data lives inside WSL** (`~/data/flight-delay`).
`scripts/bench_fs.sh` measured the reason: writing 300 small files takes 19 ms
on the WSL filesystem versus 851-1470 ms through `/mnt/c`. OneDrive itself is
not the bottleneck — the Windows/Linux bridge is.

## Getting started

Everything runs inside WSL (Ubuntu). Java 17 and Python 3.12 install into
`$HOME`, so no `sudo` and no password are needed.

```bash
scripts/setup_wsl.sh && scripts/download_bts.sh
```

```bash
source scripts/env.sh && uv sync --all-extras --group dev && uv run flight-delay status
```

```bash
scripts/check.sh
```

## Related

[TravelPlanner](https://github.com/serpeigd/TravelPlanner) — contract-driven
travel recommender: explainable ranking, hard constraints enforced in code
before the LLM writes anything, ranking evaluation without relevance labels.

## License

MIT
