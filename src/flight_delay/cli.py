"""Command line entry point.

Subcommands are added as each phase lands; ``status`` exists from the first
commit so there is always a cheap way to check what the data lake actually
contains rather than assuming.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

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

    return 0 if len(present) == len(expected) and not partial else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flight-delay", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="report what the data lake actually contains")

    args = parser.parse_args(argv)
    paths = Paths.from_env()

    if args.command == "status":
        return _cmd_status(paths)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
