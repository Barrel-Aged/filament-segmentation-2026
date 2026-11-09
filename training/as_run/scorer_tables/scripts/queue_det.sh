#!/usr/bin/env bash
# Usage: queue_det.sh <tag> <folds e.g. "0 1 2 3"> [extra train args]
# Trains Mask R-CNN fold models and writes OOF candidate roots (ext.py layout) for each held-out fold.
set -uo pipefail
cd /home/barrelaged/projects/filament-claude
TAG=$1; FOLDS=$2; shift 2
PY=.venv/bin/python
for F in $FOLDS; do
  N=${TAG}-f${F}
  if [ ! -f runs/$N/done.json ]; then
    $PY claude/src/cfil/det.py train --name $N --fold $F "$@" > logs/$N.train.log 2>&1 || { echo "train failed $N"; exit 1; }
  fi
  if [ ! -f runs/$N/oof/manifest.json ]; then
    $PY claude/src/cfil/det.py infer --name $N --split $F --out runs/$N/oof > logs/$N.infer.log 2>&1 || { echo "infer failed $N"; exit 1; }
  fi
done
echo "queue $TAG complete"
