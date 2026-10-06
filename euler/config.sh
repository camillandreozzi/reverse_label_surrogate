# Shared settings for the Euler workflow. Sourced both locally (push/pull) and on Euler (setup/submit/jobs).
#
# Workflow:
#   local:  bash euler/push.sh                         code + data -> $SCRATCH/reverse_label_surrogate
#   euler:  cd $SCRATCH/reverse_label_surrogate
#           bash euler/setup_env.sh                    once: venv + ar1_mf check
#           bash euler/submit_full_fit.sh [8]          all outputs, sf + mf (or e.g. only index 8 = f, to time one fit)
#           bash euler/submit_loo.sh                   nested LOO: 97 folds x 9 outputs = 873 tasks, independent of the full fit
#   local:  bash euler/pull.sh                         results/independent/ -> local repo
# Scratch is purged after ~2 weeks, so pull results back once jobs finish.

EULER_HOST=euler                      # ~/.ssh/config alias (euler.ethz.ch, user candreozzi)
PROJECT=reverse_label_surrogate       # directory under $SCRATCH
VENV_DIR='$HOME/venvs/rls'            # outside scratch so it survives purges (expanded on Euler)
MODULES="stack/2024-06 python/3.11.6"

VARIANTS="mf sf"                      # both fitted in every task with the same mf-tuned round count
TARGETS=("Kzz" "Rp" "Tint" "C/O" "[N/H]" "[O/H]" "[S/H]" "logg" "f")
N_HF=97

# Slurm resources (adjust after timing one full-size fit); each can be overridden per submission,
# e.g. CPUS=4 TIME_LOO=24:00:00 bash euler/submit_loo.sh 370
CPUS=${CPUS:-1}                                # 1 thread per task, parallelism from many tasks (as in Model 3)
MEM_PER_CPU=${MEM_PER_CPU:-2G}                        # observed peak ~0.6 GB per task
TIME_FULL=${TIME_FULL:-24:00:00}                    # tuning ~1.5-2 h + sf/mf fits on all rows, single thread
TIME_LOO=${TIME_LOO:-12:00:00}                     # one (fold, output) pair: tuning ~1.5-2 h + 2 fits
