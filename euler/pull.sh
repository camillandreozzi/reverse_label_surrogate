#!/usr/bin/env bash
# Local: copy results/independent/ from Euler scratch into the local repo (existing local files are kept).
set -euo pipefail
cd "$(dirname "$0")/.."
source euler/config.sh

src="$(ssh "$EULER_HOST" 'echo $SCRATCH')/$PROJECT/results/independent/"
mkdir -p results/independent
rsync -av "$EULER_HOST:$src" results/independent/
echo "Pulled $EULER_HOST:$src -> results/independent/"
