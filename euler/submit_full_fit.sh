#!/usr/bin/env bash
# Euler: submit the full fit as a job array (one task per output: shared tuning, then sf and mf) plus a merge job.
# Usage: bash euler/submit_full_fit.sh [array spec, default 0-8]   e.g. "8" = only f, to time one fit
set -euo pipefail
cd "$(dirname "$0")/.."
source euler/config.sh
array="${1:-0-$(( ${#TARGETS[@]} - 1 ))}"
logs="results/independent/logs"
mkdir -p "$logs"

jid=$(sbatch --parsable --job-name="full_fit" --array="$array" \
    --cpus-per-task="$CPUS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME_FULL" \
    --output="$logs/full_fit_%A_%a.out" euler/job_full_fit.sh)
mid=$(sbatch --parsable --job-name="ff_merge" --dependency="afterany:$jid" \
    --cpus-per-task=1 --mem-per-cpu=2G --time=00:15:00 \
    --output="$logs/full_fit_merge_%j.out" euler/job_merge.sh full_fit_model)
echo "$jid" > "$logs/full_fit.jobid"
echo "full fit array $jid (tasks $array), merge $mid; logs in $logs/"
