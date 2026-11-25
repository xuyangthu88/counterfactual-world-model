#!/usr/bin/env bash
# Run from the repository root after the Stage 22 prerequisites.
set -euo pipefail
GPU=${GPU:-0}
if [[ "$GPU" == "4" ]]; then exit 2; fi
workers=${WORKERS:-64}
sim_py=${SIM_PY:-/tmp/whatif-push-venv/bin/python}
train_py=${TRAIN_PY:-work/stage7/trainenv/bin/python}
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 OPENBLAS_NUM_THREADS=1
sim() {
    CUDA_VISIBLE_DEVICES="" MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" \
        "$sim_py" -m "whatif.feedback.$1" "${@:2}" --workers "$workers"
}
train() {
    CUDA_VISIBLE_DEVICES="$GPU" MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD" \
        "$train_py" -m "whatif.feedback.$1" "${@:2}"
}
case "${1:-results}" in
    collect)
        sim native bank
        sim native collect
        sim native expand
        sim native panels
        sim history collect
        sim native cophy
        ;;
    fit)
        train fit
        train fit --history
        ;;
    control)
        train select
        sim native execute
        train select --history
        sim history execute
        train reference
        ;;
    results)
        train metrics short
        train metrics long
        train metrics history
        train summary
        ;;
    *) exit 2 ;;
esac
