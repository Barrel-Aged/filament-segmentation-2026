#!/usr/bin/env bash
# Detector fold-k evaluation: fetch OOF root, stacking features for cnx candidates, paired test on that fold
# (baseline: cnx candidates + Codex I6 OOF greedy-set stacking + rich).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 NPROC=${NPROC:-6}
P=.venv-cpu/Scripts/python.exe
K=${1:-0}
[ -f work/det/det-f$K/oof/manifest.json ] || bash scripts/fetch_det.sh det-f$K oof
$P src/cfil/ext.py work/maps/cnx-f$K/oof.npy work/ext/cnx-det work/det/det-f$K/oof >/dev/null
$P src/cfil/rich.py work/cands/cnx work/ext/cnx-det work/feat/cnx-detrich
[ "${2:-}" = noexp ] && exit 0
$P work/exp_feat.py --only=$K --seeds=6 \
  --arm=cnx:work/cands/cnx:work/ext/cnx-i6g:work/feat/cnx-i6g-rich \
  --arm=cnx+det:work/cands/cnx:work/ext/cnx-i6g:work/feat/cnx-i6g-rich+work/ext/cnx-det+work/feat/cnx-detrich 2>&1 | grep -v Warning
