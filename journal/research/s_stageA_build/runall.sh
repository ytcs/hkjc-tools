#!/usr/bin/env bash
# runall.sh TAG CELL/KEY[:stages] ... : run1.py on each staged cell in sequence (one slot), logs to
# data_cache/s_stageA/logs/TAG/ (stages: comma list, default identify,fuse_video,fuse_final,validate)
cd "$(dirname "$0")/../../.." || exit 1
TAG=$1; shift
mkdir -p data_cache/s_stageA/logs/$TAG
for spec in "$@"; do
  ck=${spec%%:*}; st=""; [[ "$spec" == *:* ]] && st=${spec#*:}
  c=${ck%%/*}; k=${ck##*/}; log=data_cache/s_stageA/logs/$TAG/${c}_$k${st:+_${st//,/-}}.log
  uv run python journal/research/s_stageA_build/run1.py "$k" "$c" $st > "$log" 2>&1
  echo "$spec rc=$? $(tail -1 "$log")"
done
