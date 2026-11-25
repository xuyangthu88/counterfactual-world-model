#!/usr/bin/env bash
# Run from the repository root with the Stage 19 simulator and training environments.
set -euo pipefail
: "${SIM_PY:=/tmp/whatif-push-venv/bin/python}"
: "${TRAIN_PY:=work/stage7/trainenv/bin/python}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${DEVICE:=cuda}"
: "${WORKERS:=24}"
export CUDA_VISIBLE_DEVICES OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
upstream="$PWD/work/stage7/upstream"
sim() { PYTHONPATH="$upstream/pusht:$PWD" "$SIM_PY" -m "whatif.paired.$1" "${@:2}"; }
train() { PYTHONPATH="$upstream/mbrl:$PWD" "$TRAIN_PY" -m "whatif.paired.$1" "${@:2}"; }
native=(--sim-python "$SIM_PY" --workers "$WORKERS")
case "${1:-results}" in
    data)
        for domain in cophy causalworld pusht; do sim data "$domain" --workers "$WORKERS"; done
        ;;
    worlds)
        train fit --device "$DEVICE"
        train fit --device "$DEVICE" --contact-only
        ;;
    step)
        train decision --device "$DEVICE"
        train control --group previous "${native[@]}"
        train control --group new "${native[@]}"
        train option validation "${native[@]}"
        train option test "${native[@]}"
        ;;
    segments)
        mkdir -p work/mechanism
        if [[ ! -f work/mechanism/pusht-demos.zip ]]; then
            curl --fail --show-error --location https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip --output work/mechanism/pusht-demos.zip
        fi
        train segments bank
        for domain in causalworld pusht; do sim segments "$domain" --workers "$WORKERS"; done
        train decision --segments --device "$DEVICE"
        train segment_control "${native[@]}"
        ;;
    results)
        train evaluate
        train results
        ;;
    *) echo "Usage: bash whatif/paired/reproduce.sh {data|worlds|step|segments|results}" >&2; exit 2 ;;
esac
