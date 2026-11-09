#!/usr/bin/env bash
# Test + fold-4 holdout inputs for a dense source TAG (cnx | cnxL): fetch 4-model maps, candidates, I6 stacking (greedy set)
# + rich, and per-detector-fold stacking + rich (one variant per det fold model).  usage: run_final_feats.sh TAG [nodet]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 NPROC=${NPROC:-4}
P=.venv-cpu/Scripts/python.exe
T=$1
I6T=../outputs/improvement-i6/candidates/test
I6O=../outputs/improvement-i6/candidates/oof-m0,../outputs/improvement-i6/candidates/oof-m1
for s in test hold; do [ -f work/maps/final/$s$T.npy.json ] || bash scripts/fetch_maps.sh final $s$T.npy; done
$P src/cfil/build_cands.py work/maps/final/test$T.npy work/cands/test$T --test >/dev/null
$P src/cfil/build_cands.py work/maps/final/hold$T.npy work/cands/hold$T >/dev/null
$P src/cfil/ext.py work/maps/final/test$T.npy work/ext/test$T-i6g $I6T --test --greedy-sel >/dev/null
$P src/cfil/ext.py work/maps/final/hold$T.npy work/ext/hold$T-i6g $I6O --greedy-sel >/dev/null
for s in test hold; do $P src/cfil/rich.py work/cands/$s$T work/ext/$s$T-i6g work/feat/$s$T-i6g-rich; done
[ "${2:-}" = nodet ] && exit 0
for k in 0 1 2 3; do
  for s in test hold; do
    [ -f work/det/det-f$k/$s/manifest.json ] || bash scripts/fetch_det.sh det-f$k $s
    $P src/cfil/ext.py work/maps/final/$s$T.npy work/ext/$s$T-det-f$k work/det/det-f$k/$s $([ $s = test ] && echo --test) >/dev/null
    $P src/cfil/rich.py work/cands/$s$T work/ext/$s$T-det-f$k work/feat/$s$T-detrich-f$k
  done
done
echo "final feats $T done"
