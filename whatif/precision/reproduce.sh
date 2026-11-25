#!/usr/bin/env bash
set -euo pipefail
sim_py=${SIM_PY:-/tmp/whatif-push-venv/bin/python}
train_py=${TRAIN_PY:-"$PWD/work/stage7/trainenv/bin/python"}
gpu_graph=${GRAPH_GPU:-0}
gpu_baseline=${BASELINE_GPU:-5}
workers=${WORKERS:-6}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
mkdir -p work/stage8 work/stage12 results/precision data/precision reports/stages/assets
sim() { PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" "$sim_py" -m "whatif.precision.$1" "${@:2}"; }
trn() { PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD" "$train_py" -m "whatif.precision.$1" "${@:2}"; }
case "${1:-figures}" in
  collect)
    for split in train val test; do
      tasks=(causalworld pusht near)
      if [[ $split != train ]]; then tasks+=(wall); fi
      if [[ $split == test ]]; then tasks+=(p9); fi
      for task in "${tasks[@]}"; do sim collect "$task" "$split" --workers "$workers"; done
    done
    sim reference ;;
  diagnose) CUDA_VISIBLE_DEVICES=$gpu_graph trn evaluate val ;;
  train)
    for method in graph pets cophy; do
      dev=$gpu_baseline
      if [[ $method == graph ]]; then dev=$gpu_graph; fi
      CUDA_VISIBLE_DEVICES=$dev trn train "$method"
    done ;;
  select)
    CUDA_VISIBLE_DEVICES=$gpu_graph trn evaluate val --refined
    for method in graph pets cophy; do
      modes=(raw endonly exact)
      if [[ $method != cophy ]]; then modes+=(local); fi
      for mode in "${modes[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu_graph trn control controlval pusht "$method" "$mode" --count 64 --sim-python "$sim_py" --workers "$workers"
      done
    done
    trn freeze ;;
  test)
    CUDA_VISIBLE_DEVICES=$gpu_graph trn evaluate test
    CUDA_VISIBLE_DEVICES=$gpu_graph trn evaluate test --refined
    sim audit
    SIM_PY="$sim_py" trn test_control --graph-gpu "$gpu_graph" --baseline-gpu "$gpu_baseline" --workers "${PARALLEL_JOBS:-4}"
    trn summarize ;;
  figures) trn summarize ;;
  *) echo 'Expected collect, diagnose, train, select, test, or figures' >&2; exit 2 ;;
esac
