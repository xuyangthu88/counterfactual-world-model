#!/usr/bin/env bash
# Run from the shared Stage 25 repository with both earlier runtime environments available.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SIM_PY="${SIM_PY:-/tmp/whatif-push-venv/bin/python}"
train_py="${TRAIN_PY:-work/stage7/trainenv/bin/python}"
gpu="${GPU:-5}"
workers="${WORKERS:-56}"
[[ "$gpu" =~ ^[0-9]+$ && "$gpu" != 4 ]]
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 OPENBLAS_NUM_THREADS=1
sim() {
    CUDA_VISIBLE_DEVICES="" MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" \
      "$SIM_PY" -m "whatif.relay.$1" "${@:2}" --workers "$workers"
}
train() {
    CUDA_VISIBLE_DEVICES="$gpu" MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH="$PWD/work/stage7/upstream/mbrl:$PWD" \
      "$train_py" -m "whatif.relay.$1" "${@:2}"
}
closed_loop() {
    sim native initial "$@"
    train control first "$@"
    sim native observe "$@"
    train control second "$@"
    sim native execute "$@"
}
case "${1:-results}" in
    data)
        for job in cophy pusht causalworld; do sim native "$job"; done
        ;;
    world)
        for kind in dynamic fixed unpaired; do train fit "$kind"; done
        train policy
        ;;
    control)
        closed_loop --validation
        train control select
        closed_loop
        train reference
        ;;
    results)
        train temporal
        CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/work/stage7/upstream/pusht:$PWD" "$SIM_PY" -m whatif.relay.example_trace
        train atomic
        train forecast
        train results
        ;;
    physical)
        sim extra
        for kind in dynamic fixed; do train readout "$kind"; done
        train atomic --extra
        ;;
    recourse)
        train backup_data choose
        sim backup_data measure
        sim backup_data hold
        train backup world
        train backup policy
        sim backup_data prepare
        train backup_data choose --test
        sim backup_data measure --test
        sim backup_data hold --test
        closed_loop --followup
        train backup_eval
        sim recourse_trace
        train final_eval
        ;;
    *) exit 2 ;;
esac
