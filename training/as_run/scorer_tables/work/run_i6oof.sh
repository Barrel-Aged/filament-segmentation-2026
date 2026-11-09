#!/usr/bin/env bash
# cnx dev candidates: stacking features from Codex I6 group-clean OOF roots and from I5 cells, both with the greedy selected set.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 NPROC=${NPROC:-4}
P=.venv-cpu/Scripts/python.exe
I6=../outputs/improvement-i6/candidates/oof-m0,../outputs/improvement-i6/candidates/oof-m1
for k in 0 1 2 3; do
  $P src/cfil/ext.py work/maps/cnx-f$k/oof.npy work/ext/cnx-i6g $I6 --greedy-sel >/dev/null
  $P src/cfil/ext.py work/maps/cnx-f$k/oof.npy work/ext/cnx-i5g ../outputs/improvement-i5/f$k-s20261014/candidates/outer --greedy-sel >/dev/null
  echo "fold $k ready"
done
for s in i6g i5g; do $P src/cfil/rich.py work/cands/cnx work/ext/cnx-$s work/feat/cnx-$s-rich; done
$P work/exp_feat.py --seeds=5 \
  --arm=i5:work/cands/cnx:work/ext/cnx:work/feat/cnx-rich \
  --arm=i5greedy:work/cands/cnx:work/ext/cnx-i5g:work/feat/cnx-i5g-rich \
  --arm=i6oof:work/cands/cnx:work/ext/cnx-i6g:work/feat/cnx-i6g-rich 2>&1 | grep -v Warning
