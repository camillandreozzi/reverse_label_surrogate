#!/usr/bin/env bash
# Local: copy code and data to $SCRATCH/<PROJECT> on Euler (results are never pushed).
set -euo pipefail
cd "$(dirname "$0")/.."
source euler/config.sh

dest="$(ssh "$EULER_HOST" 'echo $SCRATCH')/$PROJECT"
ssh "$EULER_HOST" "mkdir -p '$dest'"
rsync -av --exclude '__pycache__' --exclude '*.pyc' \
    src modelling euler data requirements.txt "$EULER_HOST:$dest/"
echo "Pushed to $EULER_HOST:$dest"
