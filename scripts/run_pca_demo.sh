#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
exec "$PYTHON_BIN" -m lingbot_vision.pca_demo "$@"
