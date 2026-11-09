#!/usr/bin/env bash
# cnxL-f3 hit CUDA OOM while detector inference held the GPU: retrain it once det inference is done, then make the
# cnxL test + fold-4 holdout maps that queue_final2 skips after that failure.
cd /home/barrelaged/projects/filament-claude
PY=.venv/bin/python
while [ ! -f runs/det-f3/hold/manifest.json ]; do sleep 60; done
sleep 60  # let detector inference release its memory
bash claude/scripts/queue_dense.sh cnxL unet tu-convnext_tiny 48000 --workers 5 || exit 1
for s in test 4; do
  o=$([ $s = test ] && echo testcnxL || echo holdcnxL)
  [ -f runs/final/$o.npy.json ] || $PY claude/src/cfil/dense.py infer --models cnxL-f0,cnxL-f1,cnxL-f2,cnxL-f3 --split $s --out runs/final/$o.npy > logs/final-$o.log 2>&1 || echo "infer failed $o"
done
echo "cnxL resume complete"
