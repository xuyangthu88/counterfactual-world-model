#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD"
PY="work/stage7/trainenv/bin/python"
GPU="${GPU:-9}"

cmd="${1:-help}"
shift || true

case "$cmd" in
  baselines)
    dom="${1:?domain}"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m whatif.geometry.baselines cophynet_geo "$dom"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m whatif.geometry.baselines pets_geo "$dom"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m whatif.geometry.baselines ncii_geo "$dom"
    ;;
  fit)
    kind="${1:?config}"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m whatif.geometry.fit "$kind"
    ;;
  calibrate)
    kind="${1:?config}"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m whatif.geometry.fit "$kind" --calibrate
    ;;
  evaluate)
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m whatif.geometry.evaluate "$@"
    ;;
  *)
    echo "Usage: GPU=<id> bash whatif/geometry/reproduce.sh {baselines <dom>|fit <kind>|calibrate <kind>|evaluate}"
    exit 1
    ;;
esac
