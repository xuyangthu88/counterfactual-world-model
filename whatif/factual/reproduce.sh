#!/usr/bin/env bash
set -euo pipefail
SIM_PY=${SIM_PY:-/tmp/whatif-push-venv/bin/python}
TRAIN_PY=${TRAIN_PY:-"$PWD/work/stage7/trainenv/bin/python"}
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-8}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
mkdir -p work/stage8 results/factual data/factual reports/stages/assets
trn() { PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD" "$TRAIN_PY" -m "whatif.factual.$1" "${@:2}"; }
sim() { PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" "$SIM_PY" -m whatif.factual.collect "$@"; }
common=(--device "$DEVICE" --workers "$WORKERS" --sim-python "$SIM_PY")
case "${1:-figures}" in
  figures) trn history analyze --device cpu; trn summarize; trn example ;;
  collect)
    for t in causalworld pusht wall; do
      for s in train val test; do sim "$t" "$s" --workers "$WORKERS"; done
    done
    for s in val test; do sim wall_crossed "$s" --workers "$WORKERS"; done
    PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" "$SIM_PY" -m whatif.factual.history collect --workers "$WORKERS" ;;
  train)
    for m in g12 g32 cophy ncii pets; do trn train "$m" --device "$DEVICE"; done
    trn select graph ;;
  select)
    trn evaluate val --device "$DEVICE"
    trn evaluate val --crossed --device "$DEVICE"
    for t in causalworld pusht; do
      for m in cem enumerate root pets pets_plain cophy cophy_high; do trn control controlval "$t" "$m" --count 64 "${common[@]}"; done
    done
    trn select control ;;
  test)
    trn evaluate test --device "$DEVICE"
    trn evaluate test --crossed --device "$DEVICE"
    for t in causalworld pusht; do
      methods=(ours uniform static legacy cophy pets physics)
      if [[ $t == pusht ]]; then methods+=(masked fixed); fi
      for m in "${methods[@]}"; do trn control control "$t" "$m" --count 160 "${common[@]}"; done
    done
    trn history analyze --device cpu
    trn summarize
    trn example ;;
  *) echo 'Expected figures, collect, train, select or test' >&2; exit 2 ;;
esac
