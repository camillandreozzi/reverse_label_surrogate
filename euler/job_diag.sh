#!/usr/bin/env bash
# Slurm array task for modelling/independent/diag_fixed_cov.py (target passed as DIAG_TARGET).
source euler/job_env.sh
echo "fixed-cov diagnostic: target=$DIAG_TARGET task=$SLURM_ARRAY_TASK_ID cpus=$OMP_NUM_THREADS"
python modelling/independent/diag_fixed_cov.py --target "$DIAG_TARGET" --task-id "$SLURM_ARRAY_TASK_ID"
