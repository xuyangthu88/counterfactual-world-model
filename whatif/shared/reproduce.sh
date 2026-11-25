#!/usr/bin/env bash
# Run in the repository root with the pinned Stage 07 simulator and model environments.
set -euo pipefail
SIM_PY=${SIM_PY:-/tmp/whatif-push-venv/bin/python}
TRAIN_PY=${TRAIN_PY:-"$PWD/work/stage7/trainenv/bin/python"}
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-12}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
mkdir -p work/stage8 data/shared results/shared
trn() { PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD" "$TRAIN_PY" -m "whatif.shared.$1" "${@:2}"; }
sim() { PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" "$SIM_PY" -m "whatif.shared.$1" "${@:2}"; }
common=(--device "$DEVICE" --workers "$WORKERS" --sim-python "$SIM_PY")
case "${1:-figures}" in
  figures) trn summarize ;;
  collect)
    for task in causalworld pusht; do
      if [[ $task == pusht ]]; then sizes=(1680 240 256 6400 864); else sizes=(320 112 192 3200 480); fi
      sim collect "$task" train "${sizes[0]}" --workers "$WORKERS"
      sim collect "$task" val "${sizes[1]}" --workers "$WORKERS"
      sim collect "$task" test "${sizes[2]}" --workers "$WORKERS"
      sim returns "$task" train "${sizes[3]}" --workers "$WORKERS"
      sim returns "$task" val "${sizes[4]}" --workers "$WORKERS"
    done ;;
  train)
    for model in graph cophy pets ncii; do trn train "$model" --device "$DEVICE"; done
    trn value --device "$DEVICE"
    trn evaluate calibrate --device "$DEVICE" ;;
  select)
    for task in causalworld pusht; do
      trn control controlval "val-$task" --tasks "$task" --policies graph pets cophy physics cemgraph returngraph returnpets --count 48 "${common[@]}"
    done
    trn select shortlist
    for task in causalworld pusht; do
      trn control transfer_val "confirm-$task" --tasks "$task" --count 64 "${common[@]}"
    done
    trn select freeze ;;
  test)
    trn evaluate test --device "$DEVICE"
    trn control control test-causalworld --tasks causalworld --policies graph pets cophy physics previous prior poseonly --count 128 "${common[@]}"
    trn control control test-pusht --tasks pusht --policies graph pets cophy physics prior poseonly --count 128 "${common[@]}"
    sim previous --count 128
    trn archive
    trn summarize ;;
  *) echo "Expected figures, collect, train, select or test" >&2; exit 2 ;;
esac
