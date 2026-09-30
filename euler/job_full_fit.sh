#!/usr/bin/env bash
# Slurm array task: shared tuning + sf and mf full fits of one output (SLURM_ARRAY_TASK_ID = index into TARGETS).
source euler/job_env.sh
target="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "full fit: variants=$VARIANTS target=$target cpus=$OMP_NUM_THREADS"
python modelling/independent/full_fit_model.py --variants $VARIANTS --targets "$target"
