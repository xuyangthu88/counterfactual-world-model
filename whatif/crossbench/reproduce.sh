#!/usr/bin/env bash
# Run from the repository root. Supply two interpreters with the packages in Stage 07.
set -euo pipefail
: "${SIM_PY:?Set SIM_PY to the Python 3.10 simulator interpreter}"
: "${TRAIN_PY:?Set TRAIN_PY to the PyTorch interpreter}"
DEVICE=${DEVICE:-cpu}
WORKERS=${WORKERS:-12}
UPSTREAM="$PWD/work/stage7/upstream"
mkdir -p "$UPSTREAM"
fetch() {
    local project=$1 url=$2 revision=$3
    if [[ ! -d "$UPSTREAM/$project/.git" ]]; then
        git clone -q "$url" "$UPSTREAM/$project"
        git -C "$UPSTREAM/$project" checkout -q "$revision"
    fi
    if [[ $(git -C "$UPSTREAM/$project" rev-parse HEAD) != "$revision" ]]; then
        echo "Expected $project at $revision in $UPSTREAM" >&2
        exit 1
    fi
}
fetch pusht https://github.com/real-stanford/diffusion_policy.git 5ba07ac6661db573af695b419a7947ecb704690f
fetch cophy https://github.com/fabienbaradel/cophy.git 170fe463c9e80ddf0978110438a710cabb4997e2
fetch mbrl https://github.com/facebookresearch/mbrl-lib.git 3f93cccfc8d635f74e335a2f07aab6e9a48fc021
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
trn() { PYTHONPATH="$UPSTREAM/mbrl:$PWD" "$TRAIN_PY" -m "whatif.crossbench.$1" "${@:2}"; }
sim() { PYTHONPATH="$UPSTREAM/pusht:$PWD" "$SIM_PY" -m "whatif.crossbench.$1" "${@:2}"; }
case "${1:-figures}" in
    figures)
        trn summarize
        trn surface_results
        ;;
    collect)
        sim collect --task causalworld --workers "$WORKERS"
        sim collect --task pusht --splits train val test neartrain nearval --workers "$WORKERS"
        sim surface_data --tracked --split train --count 900 --workers "$WORKERS"
        sim surface_data --tracked --split val --count 150 --workers "$WORKERS"
        ;;
    train)
        trn train --device "$DEVICE"
        trn value --device "$DEVICE"
        trn surface_fit --device "$DEVICE" --name tracked
        ;;
    evaluate)
        trn evaluate --device "$DEVICE"
        common=(--device "$DEVICE" --sim-python "$SIM_PY" --workers "$WORKERS")
        trn control "${common[@]}" --split controlval --policies ours value0.4 value0.8 --layouts 32 --output value-validation.csv
        selected=$(trn select_value)
        trn control "${common[@]}" --layouts 64
        trn control "${common[@]}" --layouts 64 --policies "$selected" --output value-test.csv
        sim feedback --matched --split controlval --count 64 --workers "$WORKERS"
        sim feedback --matched --split control --count 96 --workers "$WORKERS"
        trn control "${common[@]}" --tasks pusht --start 151000 --layouts 96 --policies "$selected" pets --output coarse-confirm.csv
        trn summarize
        trn surface_results
        ;;
    *) echo "Use figures, collect, train or evaluate" >&2; exit 1 ;;
esac
