"""Measured comparisons: partition pruning, Z-ordering, Spark against DuckDB.

Claiming a layout optimisation without numbers is not worth much. Everything
in here reports files read and bytes read, not just wall-clock, because
wall-clock alone cannot tell "the filter pruned partitions" apart from "the
page cache was warm".
"""
