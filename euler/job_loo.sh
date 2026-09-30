#!/usr/bin/env bash
# Slurm array task: one (held-out HF sample, output) pair of the nested LOO.
# task id = fold * 9 + output index; tunes on the 96 other HF rows, then fits and tests mf and sf.
source euler/job_env.sh
echo "nested LOO: task $SLURM_ARRAY_TASK_ID variants=$VARIANTS cpus=$OMP_NUM_THREADS"
python modelling/independent/cv_loo.py --variants $VARIANTS --task-id "$SLURM_ARRAY_TASK_ID"
