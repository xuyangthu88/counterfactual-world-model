#!/usr/bin/env bash
# Run at repository root with the Stage 24 data, upstream installations and weights present.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SIM_PY="${SIM_PY:-/tmp/whatif-push-venv/bin/python}"
train_py="${TRAIN_PY:-work/stage7/trainenv/bin/python}"
gpu="${GPU:-0}"
workers="${WORKERS:-56}"
[[ "$gpu" =~ ^[0-9]+$ && "$gpu" != 4 ]]
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 OPENBLAS_NUM_THREADS=1
sim() {
    CUDA_VISIBLE_DEVICES="" MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" \
      "$SIM_PY" -m whatif.hitting.native "$1" --workers "$workers"
}
train() {
    CUDA_VISIBLE_DEVICES="$gpu" MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD" \
      "$train_py" -m "whatif.hitting.$1" "${@:2}"
}
case "${1:-results}" in
    collect)
        for job in cophy pusht causalworld prepare contact; do sim "$job"; done
        ;;
    world)
        train physical
        for kind in joint marginal; do train fit "$kind"; done
        train policy
        train continuations
        sim targets
        ;;
    early)
        for kind in joint marginal; do train fit "$kind" --early; done
        train policy --early
        ;;
    control)
        train control first
        sim observe
        train control second
        sim execute
        train reference
        ;;
    evidence)
        train evidence
        for job in initial mid execute; do
            CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" \
              OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$SIM_PY" -m whatif.hitting.evidence_native "$job" --workers "$workers"
            case "$job" in initial) train evidence_control first ;; mid) train evidence_control second ;; esac
        done
        train evidence_control accuracy
        train evidence_results
        ;;
    results)
        train graph_eval
        train forecast late
        train forecast early
        train horizons
        train results
        ;;
    *) exit 2 ;;
esac
