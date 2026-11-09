#!/usr/bin/env bash
# Second-opinion features from the R34 U-Net OOF maps (a different architecture) for the cnx candidates, then a paired
# 5-seed test against the best arm: cnx (best) vs + xmR (R34) vs + xmL (cnxL, measured +0.0010) vs + xmR + xmL.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 NPROC=${NPROC:-6}
P=.venv-cpu/Scripts/python.exe
for k in 0 1 2 3; do
  $P src/cfil/xmap.py work/maps/cnx-f$k/oof.npy work/maps/r34-f$k/oof.npy work/feat/cnx-xmR >/dev/null
done
echo "xmR features: $(ls work/feat/cnx-xmR | wc -l) images"
B=work/feat/cnx-i6g-rich+work/ext/cnx-det+work/feat/cnx-detrich
$P work/exp_feat.py --seeds=5 \
  --arm=cnx:work/cands/cnx:work/ext/cnx-i6g:$B \
  --arm=cnx+xmR:work/cands/cnx:work/ext/cnx-i6g:$B+work/feat/cnx-xmR \
  --arm=cnx+xmL:work/cands/cnx:work/ext/cnx-i6g:$B+work/feat/cnx-xmL \
  --arm=cnx+xmRL:work/cands/cnx:work/ext/cnx-i6g:$B+work/feat/cnx-xmR+work/feat/cnx-xmL 2>&1 | grep -v Warning
