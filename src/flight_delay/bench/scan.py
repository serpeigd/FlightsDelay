"""Pull real scan metrics out of an executed Spark plan.

Wall-clock is not evidence. A query can be fast because the layout let it skip
99% of the data, or because the OS cached the file on the previous run. Files
read and bytes read separate the two, and they come from Spark's own scan
metrics rather than from anything this code estimates.

Getting at them has two traps:

1. ``df.count()`` builds and runs its **own** plan. Reading metrics off
   ``df._jdf.queryExecution()`` afterwards yields a plan that was never
   executed, with every metric at zero. The plan object has to be fetched
   first and then executed.
2. With adaptive query execution enabled, the root node is an
   ``AdaptiveSparkPlanExec`` whose ``children()`` is empty, and below it query
   stages hold their subtree in ``plan()`` rather than ``children()``. A plain
   children-walk finds no scan at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


class NoScanError(RuntimeError):
    """The executed plan never touched a file, so there is nothing to measure."""


@dataclass(frozen=True, slots=True)
class ScanResult:
    """What one query actually read."""

    label: str
    files_read: int
    bytes_read: int
    rows_scanned: int
    partitions_read: int
    seconds: float
    result: Any = field(default=None, compare=False)

    @property
    def megabytes_read(self) -> float:
        return self.bytes_read / 1e6

    def skipping_vs(self, baseline: ScanResult) -> float:
        """Fraction of the baseline's bytes this query avoided reading."""
        if not baseline.bytes_read:
            return 0.0
        return 1.0 - self.bytes_read / baseline.bytes_read


def _collect(node: Any, out: dict[str, int]) -> None:
    name = str(node.nodeName())

    metrics: dict[str, int] = {}
    iterator = node.metrics().iterator()
    while iterator.hasNext():
        pair = iterator.next()
        metrics[str(pair._1())] = int(pair._2().value())
    # Sum across scans: a join reads more than one input.
    if "numFiles" in metrics:
        for key, value in metrics.items():
            out[key] = out.get(key, 0) + value

    if name.startswith("AdaptiveSparkPlan"):
        _collect(node.executedPlan(), out)
        return
    if "QueryStage" in name:
        _collect(node.plan(), out)
        return

    children = node.children()
    for index in range(children.size()):
        _collect(children.apply(index), out)


def measure(label: str, df: Any) -> ScanResult:
    """Execute ``df``'s own plan and report what the scan touched.

    Pass a dataframe that already aggregates (``selectExpr("count(*)")`` and
    friends): the plan is run with ``executeCollect``, so a non-aggregated
    frame would pull every row back to the driver.
    """
    plan = df._jdf.queryExecution().executedPlan()

    start = time.perf_counter()
    rows = plan.executeCollect()
    elapsed = time.perf_counter() - start

    metrics: dict[str, int] = {}
    _collect(plan, metrics)
    if "numFiles" not in metrics:
        raise NoScanError(
            f"no scan metrics for {label!r}: the plan contains no file scan. "
            "Delta answers count(*) from per-file row counts in its transaction "
            "log without opening a single file, so a benchmark query must "
            "aggregate over a real column to force a read."
        )

    return ScanResult(
        label=label,
        files_read=metrics.get("numFiles", 0),
        bytes_read=metrics.get("filesSize", 0),
        rows_scanned=metrics.get("numOutputRows", 0),
        partitions_read=metrics.get("numPartitions", 0),
        seconds=elapsed,
        # executeCollect returns Java InternalRow objects, which are not
        # iterable from Python. Their string form is enough to assert that a
        # layout change did not alter the answer.
        result=tuple(str(row) for row in rows),
    )
