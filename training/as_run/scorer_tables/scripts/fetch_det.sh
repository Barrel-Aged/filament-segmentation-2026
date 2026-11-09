#!/usr/bin/env bash
# Fetch a detector candidate root (runs/<name>/<sub>) from the remote into work/det/<name>/<sub>.
set -euo pipefail
cd "$(dirname "$0")/.."
N=$1; S=$2
mkdir -p work/det/$N
ssh -o BatchMode=yes barrelaged@barrelagedworkflows-1 "cd /home/barrelaged/projects/filament-claude/runs/$N && tar -cf - $S" < /dev/null | tar -xf - -C work/det/$N
test -f work/det/$N/$S/manifest.json && echo "fetched $N/$S $(ls work/det/$N/$S | wc -l) files"
