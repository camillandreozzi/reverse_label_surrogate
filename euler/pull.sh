#!/usr/bin/env bash
# Local: copy results/$RUN/ (default independent) from Euler scratch into the local repo (existing local files are kept).
set -euo pipefail
cd "$(dirname "$0")/.."
source euler/config.sh

src="$(ssh "$EULER_HOST" 'echo $SCRATCH')/$PROJECT/results/$RUN/"
mkdir -p "results/$RUN"
rsync -av "$EULER_HOST:$src" "results/$RUN/"
echo "Pulled $EULER_HOST:$src -> results/$RUN/"
