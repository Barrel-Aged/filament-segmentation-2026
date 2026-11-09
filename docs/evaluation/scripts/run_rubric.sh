#!/bin/bash
# Runs rubric_dev.py on dev folds 0-3, 4 CPU processes. CPU only.
D=D:/FilamentSegmentationKaggle/claude/work/package/report_inputs
PY=D:/FilamentSegmentationKaggle/claude/.venv-cpu/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONPATH=D:/FilamentSegmentationKaggle/claude/src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for k in 0 1 2 3; do "$PY" -B $D/scripts/rubric_dev.py $k > $D/rubric/rubric_f$k.log 2>&1 & done
wait
echo "all done $(date +%T)"
