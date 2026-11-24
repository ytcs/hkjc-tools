#!/bin/sh
# s_subobs runner: 2 threads, 2 G cap, main venv with the worktree's src first, cwd = worktree root.
#   run.sh LOG SCRIPT ARGS...    (SUB=1 sets HKJC_SUBSPLIT_OBS=1; BASE=1 imports data_cache/s_subobs/base_src/src)
LOG=$1; shift
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 POLARS_MAX_THREADS=2
export HKJC_NPROC=2 HKJC_OCR_THREADS=1 PYTHONHASHSEED=0
unset HKJC_MASK HKJC_FINAL_HOLDOUT HKJC_PERTURB_SEED HKJC_LAT_METRES HKJC_CONTRACT HKJC_METRIC_W HKJC_SUBSPLIT_OBS
if [ "$SUB" = "1" ]; then export HKJC_SUBSPLIT_OBS=1; fi
WT=/home/ytc/Projects/hkjc-tools/.claude/worktrees/agent-ae0bfcc1e8dc24acb
cd $WT || exit 1
export PYTHONPATH=$WT/src
if [ "$BASE" = "1" ]; then export PYTHONPATH=$WT/data_cache/s_subobs/base_src/src; fi
MEM=2G exec /home/ytc/Projects/hkjc-tools/scripts/capped.sh /home/ytc/Projects/hkjc-tools/.venv/bin/python "$@" > "$LOG" 2>&1
