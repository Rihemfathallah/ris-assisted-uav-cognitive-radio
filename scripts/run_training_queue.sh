#!/usr/bin/env bash
# TD3 training protocol used for the reported results (Stable-Baselines3 / PyTorch).
#
# Budget = the project specification:
#   pure TD3 and residual TD3   : 5 seeds x 200k steps
#   each of the four ablations  : 3 seeds x 100k steps
#
# Seeds run as parallel single-threaded processes, at most $JOBS at a time
# (default: the number of physical cores).  Finished runs are skipped, so the
# queue is resumable after an interruption: just run it again.
#
#   ./scripts/run_training_queue.sh              # full spec budget
#   JOBS=6 ./scripts/run_training_queue.sh       # limit concurrency
#   STEPS_MAIN=50000 STEPS_ABL=25000 ./scripts/run_training_queue.sh   # shorter budget
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS="${JOBS:-$(python -c 'import os;print(os.cpu_count() or 4)')}"
STEPS_MAIN="${STEPS_MAIN:-200000}"
STEPS_ABL="${STEPS_ABL:-100000}"
SEEDS_MAIN="${SEEDS_MAIN:-0 1 2 3 4}"
SEEDS_ABL="${SEEDS_ABL:-0 1 2}"
PY="${PY:-python -W ignore}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one thread per run; parallelism is across runs

run() {   # run <tag> <seed> <steps> [extra args...]
  local tag=$1 seed=$2 steps=$3; shift 3
  if [ -f "results/runs/${tag}_N16_seed${seed}/meta.json" ]; then
    echo "skip ${tag}_N16_seed${seed}"; return
  fi
  echo "start ${tag}_N16_seed${seed} (${steps} steps)"
  $PY -m ris_cr.rl.train_td3 --N 16 --seed "$seed" --tag "$tag" --timesteps "$steps" --quiet "$@" \
    > "results/runs/${tag}_N16_seed${seed}.log" 2>&1
  echo "done  ${tag}_N16_seed${seed}"
}

throttle() {   # block until fewer than JOBS background jobs are running
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
}

mkdir -p results/runs
echo "queue: JOBS=$JOBS  main=${STEPS_MAIN} steps x [${SEEDS_MAIN}]  ablations=${STEPS_ABL} steps x [${SEEDS_ABL}]"

for s in $SEEDS_MAIN; do throttle; run td3    "$s" "$STEPS_MAIN" --action-param circular & done
for s in $SEEDS_MAIN; do throttle; run td3res "$s" "$STEPS_MAIN" --action-param residual & done
for s in $SEEDS_ABL; do throttle; run abl_linearhead    "$s" "$STEPS_ABL" --action-param linear & done
for s in $SEEDS_ABL; do throttle; run abl_gamma099      "$s" "$STEPS_ABL" --gamma 0.99 & done
for s in $SEEDS_ABL; do throttle; run abl_learnP_linpen "$s" "$STEPS_ABL" --power-mode learn --penalty linear & done
for s in $SEEDS_ABL; do throttle; run abl_learnP_dbpen  "$s" "$STEPS_ABL" --power-mode learn --penalty db & done
wait
echo QUEUE_DONE
