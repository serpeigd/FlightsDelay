"""Query engines.

The same curation is implemented twice on purpose. 13.3M rows do not need
Spark -- DuckDB handles them comfortably on one laptop -- so the honest way to
justify a distributed engine is to measure both and publish where the crossover
actually is, rather than to assert that Spark was necessary.
"""
