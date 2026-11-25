#!/usr/bin/env bash
# Run in the repository root with the Stage 22/23 assets present.
set -euo pipefail
gpu=${GPU:-0}
workers=${WORKERS:-64}
export SIM_PY=${SIM_PY:-/tmp/whatif-push-venv/bin/python}
train_py=${TRAIN_PY:-work/stage7/trainenv/bin/python}
if [[ "$gpu" == "4" ]]; then exit 2; fi
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 OPENBLAS_NUM_THREADS=1
sim() {
    CUDA_VISIBLE_DEVICES="" MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" \
      "$SIM_PY" -m "whatif.recourse.$1" "${@:2}" --workers "$workers"
}
train() {
    CUDA_VISIBLE_DEVICES="$gpu" MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD" \
      "$train_py" -m "whatif.recourse.$1" "${@:2}"
}
case "${1:-results}" in
    data)
        sim native cophy
        sim native pusht
        sim native causalworld
        sim backup_data observe
        sim anchor_native labels
        sim anchor_native observe
        ;;
    world)
        train physical
        train fit
        ;;
    features)
        train continuations
        train anchor --features-only
        ;;
    returns)
        train continuations
        sim backup_data targets
        train backup
        train anchor
        ;;
    control)
        train validation
        train control first
        sim native middle
        train control second
        sim native execute
        train reference
        train anchor_control select
        train anchor_control first
        sim anchor_native middle
        train anchor_control second
        sim anchor_native execute
        train reference --anchor
        ;;
    results)
        train graph_eval
        train accuracy long
        train accuracy backup
        train results
        ;;
    *) exit 2 ;;
esac
