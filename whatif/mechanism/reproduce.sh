#!/usr/bin/env bash
# Run from repository root. The simulator and PyTorch environments have different Python ABIs.
set -euo pipefail
: "${SIM_PY:=/tmp/whatif-push-venv/bin/python}"
: "${TRAIN_PY:=work/stage7/trainenv/bin/python}"
: "${DEVICE:=cpu}"
: "${WORKERS:=12}"
upstream="$PWD/work/stage7/upstream"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 OMP_NUM_THREADS=3 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
sim() { PYTHONPATH="$upstream/pusht:$PWD" "$SIM_PY" -m "whatif.mechanism.$1" "${@:2}"; }
nn() { PYTHONPATH="$upstream/mbrl:$PWD" "$TRAIN_PY" -m "whatif.mechanism.$1" "${@:2}"; }
collect_push() {
    for domain in causalworld pusht; do
        sim collect "$domain" --workers "$WORKERS" "$@"
    done
}
fit_predictors() {
    for name in mechanism pets cophynet; do
        nn train "$name" --device "$DEVICE" "$@"
    done
}
case "${1:-}" in
    15)
        sim collect cophy --workers "$WORKERS"
        collect_push
        sim excision
        fit_predictors
        nn train ncii --device "$DEVICE"
        nn train ncii_contact --device "$DEVICE"
        nn evaluate
        ;;
    16) nn evaluate; nn figures ;;
    17)
        nn control --sim-python "$SIM_PY" --workers "$WORKERS"
        nn figures
        ;;
    18)
        collect_push --fine
        fit_predictors --fine
        nn control --fine --sim-python "$SIM_PY" --workers "$WORKERS"
        nn fine_results
        ;;
    19)
        mkdir -p work/mechanism
        archive=work/mechanism/pusht-demos.zip
        if [[ ! -f "$archive" ]]; then
            curl --fail --show-error --location https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip --output "$archive"
        fi
        nn fit_proposal "$archive" --device "$DEVICE"
        collect_push --proposal
        fit_predictors --proposal
        nn control --proposal --sim-python "$SIM_PY" --workers "$WORKERS"
        nn fine_results --proposal
        ;;
    20)
        sim causal_data
        nn directed graph --device "$DEVICE"
        nn directed ncii --device "$DEVICE"
        nn directed evaluate
        ;;
    *) echo "Usage: bash whatif/mechanism/reproduce.sh {15|16|17|18|19|20}" >&2; exit 2 ;;
esac

