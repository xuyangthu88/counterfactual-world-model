#!/usr/bin/env bash
# Run from the repository root using the Stage 07 environments and Stage 08 checkpoints.
set -euo pipefail
SIM_PY=${SIM_PY:-/tmp/whatif-push-venv/bin/python}
TRAIN_PY=${TRAIN_PY:-"$PWD/work/stage7/trainenv/bin/python"}
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-12}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
mkdir -p work/stage8 data/prospective results/prospective reports/stages/assets
trn() { PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD" "$TRAIN_PY" -m "whatif.prospective.$1" "${@:2}"; }
sim() { PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" "$SIM_PY" -m whatif.prospective.collect "$@"; }
common=(--device "$DEVICE" --workers "$WORKERS" --sim-python "$SIM_PY")
case "${1:-figures}" in
  figures) trn summarize ;;
  collect)
    for task in causalworld pusht; do
      if [[ $task == pusht ]]; then sizes=(1100 192 256); else sizes=(450 144 192); fi
      sim "$task" fit "${sizes[0]}" --workers "$WORKERS"
      sim "$task" val "${sizes[1]}" --workers "$WORKERS"
      sim "$task" test "${sizes[2]}" --workers "$WORKERS"
    done ;;
  select)
    trn fit --device "$DEVICE"
    for task in causalworld pusht; do
      trn control controlval "v-$task" --tasks "$task" --policies mass gain pets5 pets15 --count 64 "${common[@]}"
    done
    trn freeze
    trn evaluate val --device "$DEVICE" ;;
  test)
    trn evaluate test --device "$DEVICE"
    for task in causalworld pusht; do
      methods=(ours uniform legacy pets cophy physics)
      if [[ $task == pusht ]]; then methods+=(balanced_legacy); fi
      trn control control "t-$task" --tasks "$task" --policies "${methods[@]}" --count 160 "${common[@]}"
    done
    trn summarize ;;
  *) echo 'Use figures, collect, select or test.' >&2; exit 2 ;;
esac
