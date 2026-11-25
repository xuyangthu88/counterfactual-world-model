#!/usr/bin/env bash
# From repository root, with the Stage 08 data, libraries and checkpoints.
set -euo pipefail
SIM_PY=${SIM_PY:-/tmp/whatif-push-venv/bin/python}
TRAIN_PY=${TRAIN_PY:-"$PWD/work/stage7/trainenv/bin/python"}
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-10}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
mkdir -p work/stage8 data/contact_switch results/contact_switch reports/stages/assets
trn() { PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD" "$TRAIN_PY" -m "whatif.contact_switch.$1" "${@:2}"; }
sim() { PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" "$SIM_PY" -m whatif.contact_switch.collect "$@"; }
common=(--device "$DEVICE" --workers "$WORKERS" --sim-python "$SIM_PY")
case "${1:-figures}" in
  figures) trn summarize; trn example ;;
  collect)
    for t in causalworld pusht; do
      n=192
      if [[ $t == pusht ]]; then n=256; fi
      sim "$t" val 160 --workers "$WORKERS"
      sim "$t" test "$n" --workers "$WORKERS"
    done ;;
  train) trn train --device "$DEVICE" ;;
  select)
    trn evaluate val --device "$DEVICE"
    for t in causalworld pusht; do
      for m in direct cost old; do trn control controlval "$t" "$m" --count 80 "${common[@]}"; done
    done
    trn freeze ;;
  test)
    trn evaluate test --device "$DEVICE"
    alt=$("$TRAIN_PY" -c 'import json; print("cost" if json.load(open("results/contact_switch/freeze.json"))["choice"]=="direct" else "direct")')
    for t in causalworld pusht; do
      for m in ours "$alt" uniform old cophy pets physics; do trn control control "$t" "$m" --count 160 "${common[@]}"; done
    done
    trn summarize
    trn example ;;
  *) echo 'Expected figures, collect, train, select, or test' >&2; exit 2 ;;
esac
