#!/usr/bin/env bash
# C2-plus: the C2 recipe plus second-opinion features from the R34 U-Net and cnxL maps (dev +0.0013, below the gate;
# a candidate second final only). Waits for the four-model test/holdout maps, builds the xmap features, scores the
# fold-4 holdout once at the pre-registered tau 0.30 (C2: 0.4662 / thin 0.4092), then writes the test CSV.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 PYTHONPATH=../src NPROC=${NPROC:-8}
P=.venv-cpu/Scripts/python.exe
for f in testr34 holdr34 testcnxL holdcnxL; do until [ -f work/maps/final/$f.npy.json ]; do sleep 60; done; done
$P src/cfil/xmap.py work/maps/final/holdcnx.npy work/maps/final/holdr34.npy work/feat/holdcnx-xmR >/dev/null
$P src/cfil/xmap.py work/maps/final/holdcnx.npy work/maps/final/holdcnxL.npy work/feat/holdcnx-xmL >/dev/null
$P src/cfil/xmap.py work/maps/final/testcnx.npy work/maps/final/testr34.npy work/feat/testcnx-xmR --test >/dev/null
$P src/cfil/xmap.py work/maps/final/testcnx.npy work/maps/final/testcnxL.npy work/feat/testcnx-xmL --test >/dev/null
echo "features: hold $(ls work/feat/holdcnx-xmR | wc -l)/$(ls work/feat/holdcnx-xmL | wc -l) test $(ls work/feat/testcnx-xmR | wc -l)/$(ls work/feat/testcnx-xmL | wc -l)"
DEVF=work/feat/cnx-i6g-rich,work/ext/cnx-det,work/feat/cnx-detrich,work/feat/cnx-xmR,work/feat/cnx-xmL
HF=work/feat/holdcnx-i6g-rich,work/ext/holdcnx-det-f{k},work/feat/holdcnx-detrich-f{k},work/feat/holdcnx-xmR,work/feat/holdcnx-xmL
TF=work/feat/testcnx-i6g-rich,work/ext/testcnx-det-f{k},work/feat/testcnx-detrich-f{k},work/feat/testcnx-xmR,work/feat/testcnx-xmL
echo "== holdout (fold 4), C2-plus at tau 0.30 (C2: 0.4662 / thin 0.4092)"
$P src/cfil/submit2.py --oof-cands work/cands/cnx --oof-ext work/ext/cnx-i6g --oof-feat $DEVF \
  --cands work/cands/holdcnx --ext work/ext/holdcnx-i6g --feat $HF --variants 0,1,2,3 --tau 0.30 --eval 2>&1 | grep -v Warning | head -1
echo "== test CSV"
$P src/cfil/submit2.py --oof-cands work/cands/cnx --oof-ext work/ext/cnx-i6g --oof-feat $DEVF \
  --cands work/cands/testcnx --ext work/ext/testcnx-i6g --feat $TF --variants 0,1,2,3 --tau 0.30 \
  --maps work/maps/final/testcnx.npy --out work/sub/c2plus-xmRL-t030.csv 2>&1 | grep -v Warning | tail -1
echo "c2plus complete"
