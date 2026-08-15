# Working in this repo

## Where things live

Source lives on Windows under OneDrive; **every byte of data lives inside WSL**.
Measured with `scripts/bench_fs.sh` (300 small files): 19 ms on the WSL
filesystem, 851-1470 ms through `/mnt/c`. OneDrive is not the bottleneck — the
Windows/Linux bridge is, by 50-100x.

| What | Where |
|---|---|
| Source | `/mnt/c/.../Amadeus_FlightsDelay` |
| Data lake | `~/data/flight-delay` in WSL (raw, staging, lake, mlruns, bench) |
| Virtualenv | `~/.venvs/flight-delay` — **never** `.venv` in the repo |
| Caches (ruff, mypy, pytest, pycache) | `~/.cache/flight-delay` |

`scripts/env.sh` sets all of this. Source it before anything else.

## Running commands in WSL from Windows

Use the Bash tool (Git Bash), not PowerShell, and follow this shape:

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash -lc 'cd "/mnt/c/.../Amadeus_FlightsDelay" && source scripts/env.sh && ./scripts/check.sh' | tr -d '\000'
```

Four traps, all of which have already cost time here:

1. **PowerShell truncates WSL output** at the first null byte — you get one line
   and think the command failed. Always go through Git Bash with `| tr -d '\000'`.
2. **`MSYS_NO_PATHCONV=1` is required.** Git Bash rewrites any argument that
   looks like a Unix path, so `/mnt/c/...` arrives mangled.
3. **Git Bash expands `$VAR` and `$(...)` before WSL sees them**, even inside
   single quotes. `$UV_PROJECT_ENVIRONMENT` arrives empty and `$(seq 1 40)` is a
   syntax error. Write literal paths, or put the script in a file and run that.
4. **Nested single quotes break the outer `bash -lc '...'`.** A heredoc
   containing `'skip'` silently becomes `skip`. For anything non-trivial, write
   a `.py` or `.sh` file and execute it.

`make` is **not installed** and `sudo` prompts for a password, so everything is
a shell script under `scripts/`.

## Checks

```bash
./scripts/check.sh
```

ruff, `mypy --strict`, and pytest. `spark` and `data` markers are opt-in:
`PYTEST_MARK='' ./scripts/check.sh` runs them too. Commit only in green.

**CI is stricter than a fresh local environment.** With `pyspark` absent its
imports degrade to `Any` and strict mode has nothing to check; CI installs
`--all-extras` and catches what local misses. Keep pyspark installed locally.

## Invariants

**The leakage contract in `src/flight_delay/ingest/schema.py` is the point of
the project.** Every column declares when its value becomes known
(`SCHEDULED` / `DEPARTED` / `ARRIVED` / `LABEL`). Feature sets are built by
filtering on `columns_available_at`, never by listing columns by hand. Adding a
column means classifying it. `ArrDelay`, `CarrierDelay`, `LateAircraftDelay`
and friends are recorded after the outcome and are never inputs.

Other rules carried over from the previous project:

- **Split temporally, never randomly.** Train 2023, test 2024.
- **Baselines before models.** If the model does not beat them, that is the result.
- **PR-AUC and calibration**, not accuracy — the label is 20.4% positive.
- **Do not fabricate data.** Estimates are marked as estimates.
- **State limitations before being asked.**

## Facts worth not rediscovering

- The feed has **109 named columns plus a phantom empty field**: every header
  and data line ends with a trailing comma. A positional schema that ignores it
  shifts every column by one. It is named `_trailing` and dropped.
- Clock fields are zero-padded strings (`'0030'`), numerics arrive as `'-3.00'`.
- **`delta-spark` 4.3.1 requires `pyspark >=4.0.1,<=4.1.1`.** pyspark 4.2 and
  delta-spark 4.3.1 cannot coexist.
- **`configure_spark_with_delta_pip` only adds the Maven coordinates.** Setting
  `spark.sql.extensions` and `spark.sql.catalog.spark_catalog` is the caller's
  job (see `DELTA_CONFIG` in `src/flight_delay/spark.py`). Without them the
  session starts clean and the first Delta write fails.
- `unzip` is not installed, so extraction goes through Python's `zipfile`.
- Python buffers stdout when redirected: use `flush=True` in scripts whose
  progress you intend to watch through a log file.

## Verification

Do not report that something works without having run it, and read the actual
output rather than only checking the exit code. Every significant finding so
far came from reading output: the single unlabelled flight with no arrival
record, the one-scan-per-column performance bug, and a Delta session that
looked healthy while the extension was unset.

`CONTEXTO.md` is personal interview-preparation notes. It is gitignored and
must never be committed or published.
