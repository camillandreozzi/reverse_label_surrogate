#!/usr/bin/env bash
# Euler: submit the nested 97-fold HF LOO as one job array (one task per held-out sample x output,
# 97 x 9 = 873 tasks) plus a merge job. Tuning happens inside each task, so there is no dependency on
# the full fit. Resubmitting skips (fold, output) pairs that already finished.
# Usage: bash euler/submit_loo.sh [array spec, default all 0-872]
set -euo pipefail
cd "$(dirname "$0")/.."
source euler/config.sh
logs="results/$RUN/logs"
mkdir -p "$logs"

n_tasks=$(( N_HF * ${#TARGETS[@]} ))
array="${1:-0-$(( n_tasks - 1 ))}"

jid=$(sbatch --parsable --export=ALL,RUN="$RUN" --job-name="loo" --array="$array" \
    --cpus-per-task="$CPUS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME_LOO" \
    --output="$logs/loo_%A_%a.out" euler/job_loo.sh)
mid=$(sbatch --parsable --export=ALL,RUN="$RUN" --job-name="loo_merge" --dependency="afterany:$jid" \
    --cpus-per-task=1 --mem-per-cpu=2G --time=00:15:00 \
    --output="$logs/loo_merge_%j.out" euler/job_merge.sh cv_loo)
echo "nested LOO array $jid (tasks $array of $n_tasks = $N_HF folds x ${#TARGETS[@]} outputs, {$VARIANTS} each), merge $mid"
