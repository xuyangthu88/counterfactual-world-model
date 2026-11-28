#!/usr/bin/env bash
set -euo pipefail
PY=work/stage7/trainenv/bin/python
# 1. Retrain fair original-budget baselines on universal M=2 multi-convex parts (Push-T non-convex domain)
$PY -m whatif.multiconvex.baselines all pusht
# 2. Train & calibrate Stage 35 MultiConvexDeltaGraphWorld variants
for kind in mc_shared5_hop3 mc_shared3_hop3 mc_shared5_noexo mc_unseen_pusht4 mc_unseen_stacking2; do
  $PY -m whatif.multiconvex.fit "$kind"
  $PY -m whatif.multiconvex.fit "$kind" --calibrate
done
# 3. Unified evaluation & figures
$PY -m whatif.multiconvex.evaluate
$PY -m whatif.multiconvex.figures
