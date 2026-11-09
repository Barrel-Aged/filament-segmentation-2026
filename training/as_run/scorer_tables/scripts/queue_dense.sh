#!/usr/bin/env bash
# Usage: queue_dense.sh <tag> <arch> <encoder> <steps> [extra train args]
# Trains fold models 0..3 sequentially and writes OOF maps for each held-out fold.
set -uo pipefail
cd /home/barrelaged/projects/filament-claude
TAG=$1; ARCH=$2; ENC=$3; STEPS=$4; shift 4
PY=.venv/bin/python
for F in 0 1 2 3; do
  N=${TAG}-f${F}
  if [ ! -f runs/$N/done.json ]; then
    $PY claude/src/cfil/dense.py train --name $N --fold $F --arch $ARCH --encoder $ENC --steps $STEPS "$@" > logs/$N.train.log 2>&1 || { echo "train failed $N"; exit 1; }
  fi
  if [ ! -f runs/$N/oof.npy.json ]; then
    $PY claude/src/cfil/dense.py infer --models $N --split $F --out runs/$N/oof.npy > logs/$N.infer.log 2>&1 || { echo "infer failed $N"; exit 1; }
  fi
done
echo "queue $TAG complete"
