#!/usr/bin/env bash
# Launch the longer ConvNeXt-T dense queue (48k steps) detached.
cd /home/barrelaged/projects/filament-claude
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup bash claude/scripts/queue_dense.sh cnxL unet tu-convnext_tiny 48000 --workers 5 > logs/queue-cnxL.log 2>&1 < /dev/null &
echo started
