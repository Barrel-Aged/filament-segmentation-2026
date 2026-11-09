#!/usr/bin/env bash
# Wait for det fold 0 to finish (OOF written), then train/infer det folds 1-3.
cd /home/barrelaged/projects/filament-claude
while [ ! -f runs/det-f0/oof/manifest.json ]; do
  sleep 60
  grep -q failed logs/queue-det.log 2>/dev/null && { echo "fold 0 failed; not continuing"; exit 1; }
done
bash claude/scripts/queue_det.sh det "1 2 3"
