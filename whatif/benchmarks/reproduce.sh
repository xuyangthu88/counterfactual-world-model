#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD"
PY="work/stage7/trainenv/bin/python"
GPU="${GPU:-9}"

cmd="${1:-help}"
shift || true

case "$cmd" in
  baselines)
    dom="${1:?ballscf or stacking2}"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m whatif.benchmarks.baselines cophynet "$dom"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m whatif.benchmarks.baselines pets "$dom"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m whatif.benchmarks.baselines ncii "$dom"
    ;;
  fit)
    kind="${1:?config name}"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m whatif.benchmarks.fit "$kind"
    ;;
  calibrate)
    kind="${1:?config name}"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m whatif.benchmarks.fit "$kind" --calibrate
    ;;
  evaluate)
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m whatif.benchmarks.evaluate "$@"
    "$PY" -m whatif.benchmarks.figures
    ;;
  *)
    echo "Usage: GPU=<id> bash whatif/benchmarks/reproduce.sh {baselines <dom>|fit <kind>|calibrate <kind>|evaluate}"
    exit 1
    ;;
esac
