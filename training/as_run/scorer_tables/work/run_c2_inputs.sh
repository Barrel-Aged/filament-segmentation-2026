#!/usr/bin/env bash
# C2 inputs: cnx test/holdout features now, then det fold features once every det-f{k} test+hold root exists remotely.
set -euo pipefail
cd "$(dirname "$0")/.."
bash work/run_final_feats.sh cnx nodet
echo "nodet feats done"
for k in 0 1 2 3; do for s in test hold; do
  until timeout 20 ssh -o BatchMode=yes barrelaged@barrelagedworkflows-1 "test -f /home/barrelaged/projects/filament-claude/runs/det-f$k/$s/manifest.json" < /dev/null; do sleep 90; done
done; done
bash work/run_final_feats.sh cnx
