#!/usr/bin/env bash
cd /home/barrelaged/projects/filament-claude
setsid nohup bash claude/scripts/queue_final2.sh > logs/queue-final2.log 2>&1 < /dev/null &
echo started
