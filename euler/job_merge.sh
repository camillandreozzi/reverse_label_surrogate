#!/usr/bin/env bash
# Slurm job: combine per-output / per-pair files. Usage: job_merge.sh full_fit_model|cv_loo
source euler/job_env.sh
python "modelling/independent/$1.py" --variants $VARIANTS --merge
