#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" -B smoke_test.py --backend "${1:-all}" --device "${DEVICE:-cuda:0}"
