#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-eval}
backend=${2:-all}
dataset=${3:-all}
case "$mode" in eval|train) ;; *) echo "Usage: bash scripts/table1.sh [eval|train] [all|linrec|hstu|fuxilinear|blossomrec] [all|ml20m|amazon|kuairand]" >&2; exit 2;; esac
case "$backend" in all|linrec|hstu|fuxilinear|blossomrec) ;; *) exit 2;; esac
case "$dataset" in all|ml20m|amazon|kuairand) ;; *) exit 2;; esac
for b in linrec hstu fuxilinear blossomrec; do
  [[ "$backend" == all || "$backend" == "$b" ]] || continue
  for d in ml20m amazon kuairand; do
    [[ "$dataset" == all || "$dataset" == "$d" ]] || continue
    id=${b}_${d}
    args=(--config "configs/${id}.json" --data-root "${DATA_ROOT:-data}" --output "${OUTPUT_ROOT:-outputs/table1}/${id}" --device "${DEVICE:-cuda:0}")
    if [[ "$mode" == eval ]]; then args+=(--evaluate --checkpoint "${CHECKPOINT_ROOT:-checkpoints}/${id}.pt"); fi
    "${PYTHON:-python}" "train_${b}.py" "${args[@]}"
  done
done
