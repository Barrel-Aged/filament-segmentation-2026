#!/usr/bin/env bash
# After the cnxL and det fold queues finish: test + fold-4 holdout inference for cnx, cnxL (4-model averages)
# and each det fold model. Writes into runs/final and runs/det-f*/{test,hold}.
cd /home/barrelaged/projects/filament-claude
PY=.venv/bin/python
waitfor() { while [ ! -f "$1" ]; do sleep 120; grep -q failed logs/queue-cnxL.log logs/queue-det123.log 2>/dev/null && { echo "upstream failed"; exit 1; }; done; }
run() { [ -f runs/final/$2.npy.json ] || $PY claude/src/cfil/dense.py infer --models $1 --split $3 --out runs/final/$2.npy > logs/final-$2.log 2>&1 || echo "infer failed $2"; }
# GPU memory is full while det and cnxL train together: start once the det fold queue is done
waitfor runs/det-f3/oof/manifest.json
run cnx-f0,cnx-f1,cnx-f2,cnx-f3 testcnx test
run cnx-f0,cnx-f1,cnx-f2,cnx-f3 holdcnx 4
for k in 0 1 2 3; do
  for s in test 4; do
    o=runs/det-f$k/$([ $s = test ] && echo test || echo hold)
    [ -f $o/manifest.json ] || $PY claude/src/cfil/det.py infer --name det-f$k --split $s --out $o > logs/det-f$k-$s.log 2>&1 || echo "det infer failed $k $s"
  done
done
waitfor runs/cnxL-f3/oof.npy.json
run cnxL-f0,cnxL-f1,cnxL-f2,cnxL-f3 testcnxL test
run cnxL-f0,cnxL-f1,cnxL-f2,cnxL-f3 holdcnxL 4
echo "final2 complete"
