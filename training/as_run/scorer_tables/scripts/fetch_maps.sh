#!/usr/bin/env bash
# Wait for a remote map file (its .json marks completion) and copy it into claude/work/maps/<run>/.
# usage: fetch_maps.sh <run> <file>   e.g. fetch_maps.sh r34-f0 oof.npy
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/barrelaged/projects/filament-claude/runs/$1/$2
H=barrelaged@barrelagedworkflows-1
until ssh -o BatchMode=yes $H "test -f $R.json"; do sleep 60; done
mkdir -p work/maps/$1
scp -q -o BatchMode=yes $H:$R.json work/maps/$1/$2.json
scp -q -o BatchMode=yes $H:$R work/maps/$1/$2
echo "fetched $1/$2 $(du -h work/maps/$1/$2 | cut -f1)"
