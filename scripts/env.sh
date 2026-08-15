#!/usr/bin/env bash
# Source this before anything else:  source scripts/env.sh
#
# Every cache, virtualenv and artifact directory is pushed into the WSL home so
# that OneDrive only ever sees source files. Measured cost of getting this
# wrong (scripts/bench_fs.sh, 300 small files): 19 ms on WSL vs 851-1470 ms
# through /mnt/c.

export FLIGHT_DELAY_DATA="$HOME/data/flight-delay"

# Toolchain installed without sudo by scripts/setup_wsl.sh
export JAVA_HOME="$HOME/.local/opt/jdk17"
export PATH="$JAVA_HOME/bin:$HOME/.local/bin:$PATH"

# uv must not create .venv inside the repo: it would land on /mnt/c.
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/flight-delay"

export PYTHONPYCACHEPREFIX="$HOME/.cache/flight-delay/pycache"
export RUFF_CACHE_DIR="$HOME/.cache/flight-delay/ruff"
export MYPY_CACHE_DIR="$HOME/.cache/flight-delay/mypy"
export MLFLOW_TRACKING_URI="file:$FLIGHT_DELAY_DATA/mlruns"

# Spark scratch space: shuffle spill is the heaviest writer in the project.
export SPARK_LOCAL_DIRS="$HOME/.cache/flight-delay/spark-tmp"

mkdir -p "$HOME/.cache/flight-delay" "$SPARK_LOCAL_DIRS" "$FLIGHT_DELAY_DATA"
