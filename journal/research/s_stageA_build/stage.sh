#!/usr/bin/env bash
# stage.sh KEY SRCDIR CELL : reflink a main run dir's upstream artifacts into data_cache/s_stageA/runs/CELL/KEY
# (no identify / fuse / validate outputs). tracks.npz does not depend on a legend mask.
set -e
KEY=$1; SRC=$2; CELL=$3
DST=$(dirname "$0")/../../../data_cache/s_stageA/runs/$CELL/$KEY
mkdir -p "$DST"
for f in tracks.npz legend.npz silk.npz finish.npz frames_idx.npz hud.npz start.npz film.npz rail.npz metric.npz \
         pose.npz map_meta.npz marker_profile.npz dets.npz yolo.npz numbers.npz; do
  if [ -e "$SRC/$f" ] && [ ! -e "$DST/$f" ]; then cp --reflink=auto "$SRC/$f" "$DST/$f"; fi
done
case "$CELL" in lg*) echo "legend=10@${CELL:2:1}" > "$DST/mask.txt";; esac
echo "$DST"
