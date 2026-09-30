# Sourced at the start of every Slurm job: project dir, modules, venv, thread counts.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source euler/config.sh
module load $MODULES
source "$(eval echo "$VENV_DIR")/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
