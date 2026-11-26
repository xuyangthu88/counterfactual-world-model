#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
train_py="${TRAIN_PY:-work/stage7/trainenv/bin/python}"
gpu="${GPU:-0}"
[[ "$gpu" =~ ^[0-9]+$ && "$gpu" != 4 ]]
case "${1:-evaluate}" in
    data)
        bash whatif/shortstep/reproduce.sh data "${@:2}" --split transfer --transfer-base 13600000 --transfer-name cumulativetransfer
        ;;
    fit)
        CUDA_VISIBLE_DEVICES="$gpu" "$train_py" -u -m whatif.cumulative.fit "${@:2}"
        ;;
    evaluate)
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD" \
            "$train_py" -u -m whatif.cumulative.evaluate "${@:2}"
        CUDA_VISIBLE_DEVICES="" "$train_py" -m whatif.cumulative.figures
        ;;
    *)
        exit 2
        ;;
esac
