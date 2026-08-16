"""Command line entry point.

Every number in ``docs/`` is produced by a command here. That is the point of
the file: a reader who doubts a figure can run the step that made it, rather
than taking a notebook's word for it.

The pipeline, in order:

    flight-delay status      what the data lake actually contains
    flight-delay extract     ZIPs -> CSV staging
    flight-delay curate      CSV -> Parquet, with the type contract enforced
    flight-delay features    curated -> model table, with the cutoff applied
    flight-delay train       both scenarios against their baselines
    flight-delay analyse     feature importance and threshold choice
    flight-delay calibrate   two calibration holdouts, both of which failed
    flight-delay forecast    daily delay rate, rolling-origin backtest

    flight-delay export-model    deployable bundle, registered in MLflow

    flight-delay bench-layout    partition pruning and clustering (needs Java)
    flight-delay bench-engines   DuckDB against Spark  (needs Java)

Then serve it:

    uvicorn flight_delay.serving.api:app
    streamlit run src/flight_delay/serving/dashboard.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from flight_delay.config import MONTHS, YEARS, Paths, mlflow_tracking_uri


def _cmd_status(paths: Paths) -> int:
    print(f"data root       {paths.root}")
    print(f"mlflow uri      {mlflow_tracking_uri(paths)}")

    expected = [(y, m) for y in YEARS for m in MONTHS]
    present = [(y, m) for y, m in expected if paths.raw_zip(y, m).is_file()]
    partial = sorted(paths.raw.glob("*.part")) if paths.raw.is_dir() else []

    total_bytes = sum(paths.raw_zip(y, m).stat().st_size for y, m in present)
    print(f"raw months      {len(present)}/{len(expected)}  ({total_bytes / 1e6:.0f} MB)")
    if partial:
        # A ``.part`` file is a download that never finished. Counting every
        # entry in the directory hides this, which is exactly how a truncated
        # month slipped through once already.
        print(
            f"INCOMPLETE      {len(partial)} unfinished download(s): "
            f"{', '.join(p.name for p in partial)}"
        )
    if missing := [f"{y}_{m}" for y, m in expected if (y, m) not in present]:
        print(f"missing         {', '.join(missing)}")

    for name, produced_by in (
        ("flights_duckdb", "curate"),
        ("model_table", "features"),
    ):
        table = paths.table(name)
        state = "present" if table.is_dir() else f"missing (run `flight-delay {produced_by}`)"
        print(f"{name:15s} {state}")

    return 0 if len(present) == len(expected) and not partial else 1


def _dispatch(command: str, paths: Paths, args: argparse.Namespace) -> int:
    # Imported lazily so `status` stays instant and does not need pandas,
    # sklearn or a JVM just to count files.
    if command == "status":
        return _cmd_status(paths)

    handlers: dict[str, Callable[[], int]] = {}

    if command in {"extract", "curate", "features"}:
        from flight_delay.commands import data

        handlers = {
            "extract": lambda: data.cmd_extract(paths, force=args.force),
            "curate": lambda: data.cmd_curate(paths),
            "features": lambda: data.cmd_features(paths),
        }
    elif command in {"train", "analyse", "calibrate"}:
        from flight_delay.commands import modelling

        handlers = {
            "train": lambda: modelling.cmd_train(paths),
            "analyse": lambda: modelling.cmd_analyse(paths),
            "calibrate": lambda: modelling.cmd_calibrate(paths),
        }
    elif command == "forecast":
        from flight_delay.commands import forecasting

        handlers = {"forecast": lambda: forecasting.cmd_forecast(paths)}
    elif command == "export-model":
        from flight_delay.commands import serving

        handlers = {"export-model": lambda: serving.cmd_export_model(paths)}
    elif command in {"bench-layout", "bench-engines"}:
        from flight_delay.commands import benchmarking

        handlers = {
            "bench-layout": lambda: benchmarking.cmd_bench_layout(paths),
            "bench-engines": lambda: benchmarking.cmd_bench_engines(paths),
        }

    if command not in handlers:
        raise AssertionError(f"unhandled command: {command}")
    return handlers[command]()


COMMANDS: tuple[tuple[str, str], ...] = (
    ("status", "report what the data lake actually contains"),
    ("extract", "unpack the monthly ZIPs into CSV staging"),
    ("curate", "staged CSV to partitioned Parquet, counting cast failures"),
    ("features", "build the model table with the prediction cutoff applied"),
    ("train", "both scenarios, baselines first, logged to MLflow"),
    ("analyse", "permutation importance and threshold selection"),
    ("calibrate", "compare two calibration holdouts"),
    ("forecast", "rolling-origin backtest of the daily delay rate"),
    ("export-model", "write the deployable bundle and register it in MLflow"),
    ("bench-layout", "partition pruning and clustering (needs Java)"),
    ("bench-engines", "DuckDB against Spark on identical SQL (needs Java)"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flight-delay",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in COMMANDS:
        child = sub.add_parser(name, help=help_text)
        if name == "extract":
            child.add_argument(
                "--force", action="store_true", help="re-extract months already staged"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _dispatch(args.command, Paths.from_env(), args)


if __name__ == "__main__":
    raise SystemExit(main())
