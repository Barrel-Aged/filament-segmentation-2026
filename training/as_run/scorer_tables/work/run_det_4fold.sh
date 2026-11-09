#!/usr/bin/env bash
# Accumulate det OOF stacking features for folds 1-3 as they land, then the 4-fold paired confirmation.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 NPROC=${NPROC:-4}
for k in 1 2 3; do
  until timeout 20 ssh -o BatchMode=yes barrelaged@barrelagedworkflows-1 "test -f /home/barrelaged/projects/filament-claude/runs/det-f$k/oof/manifest.json" < /dev/null; do sleep 120; done
  bash work/run_det_eval.sh $k noexp
  echo "det fold $k features ready"
done
.venv-cpu/Scripts/python.exe work/exp_feat.py --seeds=5 \
  --arm=base:work/cands/cnx:work/ext/cnx-i6g:work/feat/cnx-i6g-rich \
  --arm=+det:work/cands/cnx:work/ext/cnx-i6g:work/feat/cnx-i6g-rich+work/ext/cnx-det+work/feat/cnx-detrich 2>&1 | grep -v Warning
