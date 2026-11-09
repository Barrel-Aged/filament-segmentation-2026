#!/usr/bin/env bash
# Fetch each fold's OOF maps as they finish, build r34/cnx/ens candidates + I5 stacking features, then score 4-fold.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 NPROC=8
P=.venv-cpu/Scripts/python.exe
for k in 0 1 2 3; do
  I5=../outputs/improvement-i5/f$k-s20261014/candidates/outer
  for t in r34 cnx; do
    [ -f work/maps/$t-f$k/oof.npy.json ] || bash scripts/fetch_maps.sh $t-f$k oof.npy
    $P src/cfil/build_cands.py work/maps/$t-f$k/oof.npy work/cands/$t >/dev/null
    $P src/cfil/ext.py work/maps/$t-f$k/oof.npy work/ext/$t $I5 >/dev/null
  done
  [ -f work/maps/ens-f$k.npy.json ] || $P src/cfil/avg_maps.py work/maps/ens-f$k.npy work/maps/r34-f$k/oof.npy work/maps/cnx-f$k/oof.npy
  $P src/cfil/build_cands.py work/maps/ens-f$k.npy work/cands/ens >/dev/null
  $P src/cfil/ext.py work/maps/ens-f$k.npy work/ext/ens $I5 >/dev/null
  echo "fold $k ready"
done
for t in r34 cnx ens; do
  echo "== $t base"; $P src/cfil/decide.py work/cands/$t 2>&1 | grep -E "fixed 0.4|sweep best"
  echo "== $t +I5"; $P src/cfil/decide.py work/cands/$t --ext=work/ext/$t 2>&1 | grep -E "sweep"
done
