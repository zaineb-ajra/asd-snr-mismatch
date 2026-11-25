#!/usr/bin/env bash
# Local continuous integration: lint, formatting check, tests.
#
# Usage: scripts/ci.sh [extra pytest arguments]
# Uses the interpreter in $PYTHON, else .venv/bin/python, else `python`.
# GPU tests run on the first visible device; select it with CUDA_VISIBLE_DEVICES.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -n "${PYTHON:-}" ]]; then
    py="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
    py=.venv/bin/python
else
    py=python
fi

echo "== ruff check"
"$py" -m ruff check .
echo "== black --check"
"$py" -m black --check .
echo "== pytest"
"$py" -m pytest "$@"
echo "== CI passed"
