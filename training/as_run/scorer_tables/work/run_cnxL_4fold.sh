#!/usr/bin/env bash
# 4-fold cnxL evaluation (waits for each remote cnxL OOF map): cnxL candidates with the full feature set
# (I6 greedy + rich + det + detrich) vs cnx, and cnx candidates + cnxL second-opinion features (xmap).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 NPROC=${NPROC:-6}
P=.venv-cpu/Scripts/python.exe
I6=../outputs/improvement-i6/candidates/oof-m0,../outputs/improvement-i6/candidates/oof-m1
for k in 0 1 2 3; do
  [ -f work/maps/cnxL-f$k/oof.npy.json ] || bash scripts/fetch_maps.sh cnxL-f$k oof.npy
  $P src/cfil/build_cands.py work/maps/cnxL-f$k/oof.npy work/cands/cnxL >/dev/null
  $P src/cfil/ext.py work/maps/cnxL-f$k/oof.npy work/ext/cnxL-i6g $I6 --greedy-sel >/dev/null
  $P src/cfil/ext.py work/maps/cnxL-f$k/oof.npy work/ext/cnxL-det work/det/det-f$k/oof >/dev/null
  $P src/cfil/xmap.py work/maps/cnx-f$k/oof.npy work/maps/cnxL-f$k/oof.npy work/feat/cnx-xmL >/dev/null
  echo "cnxL fold $k features ready"
done
$P src/cfil/rich.py work/cands/cnxL work/ext/cnxL-i6g work/feat/cnxL-i6g-rich
$P src/cfil/rich.py work/cands/cnxL work/ext/cnxL-det work/feat/cnxL-detrich
B=work/feat/cnx-i6g-rich+work/ext/cnx-det+work/feat/cnx-detrich
$P work/exp_feat.py --seeds=5 \
  --arm=cnx:work/cands/cnx:work/ext/cnx-i6g:$B \
  --arm=cnxL:work/cands/cnxL:work/ext/cnxL-i6g:work/feat/cnxL-i6g-rich+work/ext/cnxL-det+work/feat/cnxL-detrich \
  --arm=cnx+xmL:work/cands/cnx:work/ext/cnx-i6g:$B+work/feat/cnx-xmL 2>&1 | grep -v Warning
