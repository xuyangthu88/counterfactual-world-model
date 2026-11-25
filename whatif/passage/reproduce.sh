#!/usr/bin/env bash
set -euo pipefail
sim_py=${SIM_PY:-/tmp/whatif-push-venv/bin/python}
train_py=${TRAIN_PY:-"$PWD/work/stage7/trainenv/bin/python"}
workers=${WORKERS:-8}
export GRAPH_GPU=${GRAPH_GPU:-0} BASELINE_GPU=${BASELINE_GPU:-5} PARALLEL_JOBS=${PARALLEL_JOBS:-4}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
mkdir -p data/passage results/passage work/stage13 work/stage8 reports/stages/assets
train() { PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD" "$train_py" -m "whatif.passage.$1" "${@:2}"; }
sim() { PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" "$sim_py" -m "whatif.passage.$1" "${@:2}"; }
suite=(--graph-gpu "$GRAPH_GPU" --baseline-gpu "$BASELINE_GPU" --parallel "$PARALLEL_JOBS")
case "${1:-figures}" in
  collect)
    for split in val test; do
      for t in pusht causalworld near finish wall p9; do sim collect "$t" "$split" --workers "$workers"; done
      for t in pusht causalworld finish; do sim repeated "$t" "$split" --workers "$workers"; done
    done
    for split in train val; do for t in pusht causalworld; do sim returns "$t" "$split" --workers "$workers"; done; done ;;
  passage)
    train history
    CUDA_VISIBLE_DEVICES=$GRAPH_GPU train train
    train duration
    CUDA_VISIBLE_DEVICES=$GRAPH_GPU train evaluate val
    train control_suite controlval "${suite[@]}"
    train freeze
    CUDA_VISIBLE_DEVICES=$GRAPH_GPU train evaluate test
    train control_suite control "${suite[@]}" ;;
  returns)
    for m in graph pets cophy; do CUDA_VISIBLE_DEVICES=$GRAPH_GPU train advantage "$m"; done
    CUDA_VISIBLE_DEVICES=$GRAPH_GPU train return_diagnosis
    train adv_suite reference "${suite[@]}"
    train adv_suite mix "${suite[@]}"
    train adv_suite freeze
    train adv_suite test "${suite[@]}" ;;
  joint)
    train joint freeze
    train joint controlval
    train joint control ;;
  figures) train summarize ;;
  *) echo 'Expected collect, passage, returns, joint or figures' >&2; exit 2 ;;
esac
