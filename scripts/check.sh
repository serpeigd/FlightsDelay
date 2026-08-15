#!/usr/bin/env bash
# Lint, type-check and test. This is what CI runs and what a commit must pass.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=./env.sh
source scripts/env.sh

BIN="$UV_PROJECT_ENVIRONMENT/bin"
if [ ! -x "$BIN/python" ]; then
  echo "No environment at $UV_PROJECT_ENVIRONMENT. Run:" >&2
  echo "  uv sync --all-extras --group dev" >&2
  exit 1
fi

echo "== ruff =="
"$BIN/ruff" check .
"$BIN/ruff" format --check .

echo "== mypy (strict) =="
"$BIN/mypy"

echo "== pytest =="
# Spark and data-lake tests are opt-in: PYTEST_MARK='' scripts/check.sh
"$BIN/pytest" -m "${PYTEST_MARK-not spark and not data}"

echo
echo "TODO EN VERDE."
