#!/usr/bin/env bash
# Euler, once: create the venv and check that this gpboost build has the AR(1) multi-fidelity kernel.
set -euo pipefail
cd "$(dirname "$0")/.."
source euler/config.sh
venv=$(eval echo "$VENV_DIR")

module load $MODULES
python -m venv "$venv"
source "$venv/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

python - <<'EOF'
import sys
import numpy as np
import gpboost as gpb
coords = np.column_stack([np.random.rand(20, 3), np.r_[np.ones(5), np.zeros(15)]])
try:
    gpb.GPModel(gp_coords=coords, cov_function="ar1_mf_matern", cov_fct_shape=1.5,
                gp_approx="vecchia_euclidean", num_neighbors=5, likelihood="gaussian")
except Exception as e:
    sys.exit(f"gpboost {gpb.__version__} lacks ar1_mf_matern / vecchia_euclidean: {e}\n"
             "Install the same gpboost build as on your laptop into this venv.")
print(f"gpboost {gpb.__version__}: ar1_mf_matern + vecchia_euclidean OK")
EOF
