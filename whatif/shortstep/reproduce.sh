#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
train_py="${TRAIN_PY:-work/stage7/trainenv/bin/python}"
sim_py="${SIM_PY:-/tmp/whatif-push-venv/bin/python}"
gpu="${GPU:-0}"
[[ "$gpu" =~ ^[0-9]+$ && "$gpu" != 4 ]]
case "${1:-evaluate}" in
    data)
        CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" \
            "$sim_py" -u -m whatif.shortstep.collect "${@:2}"
        ;;
    fit|calibrate)
        CUDA_VISIBLE_DEVICES="$gpu" "$train_py" -u -m "whatif.shortstep.$1" "${@:2}"
        ;;
    evaluate)
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD" \
            "$train_py" -u -m whatif.shortstep.evaluate "${@:2}"
        CUDA_VISIBLE_DEVICES="" "$train_py" -m whatif.shortstep.figures
        ;;
    *)
        exit 2
        ;;
esac
