#!/bin/sh
# Fast dev iteration: re-run only the given stages on a race subset, JOBS races in parallel (each
# under its own memory cap), then print the one-line gate summaries.
#   scripts/dev_loop.sh                                   # identify,fuse,validate on the dev subset
#   STAGES=fuse,validate RACES="20260603 HV 8;20260520 HV 2" scripts/dev_loop.sh
#   JOBS=4 when the machine is free (default 2: other agents share the 15 GB)
#   RACES=all STAGES=rails,observe,identify,fuse,validate scripts/dev_loop.sh   # 10-race set
cd "$(dirname "$0")/.." || exit 1
STAGES=${STAGES:-identify,fuse,validate}
DEV="20260603 HV 8;20260603 HV 3;20260520 HV 2;20260527 HV 3"
ALL="$DEV;20260429 HV 3;20260607 ST 6;20260607 ST 5;20260607 ST 8;20260503 ST 7;20260401 ST 6"
RACES=${RACES:-$DEV}
[ "$RACES" = all ] && RACES=$ALL
LOG=data_cache/recon/dev_loop
mkdir -p $LOG
J=${JOBS:-2}
HKJC_OCR_THREADS=${HKJC_OCR_THREADS:-$(( $(nproc) / J ))}  # no thread oversubscription across races
export STAGES LOG HKJC_OCR_THREADS
echo "$RACES" | tr ';' '\n' | grep . | xargs -P "${JOBS:-2}" -I{} sh -c '
  set -- {}; MEM=${MEM:-3G} scripts/capped.sh uv run --group vision python -m hkjc.recon "$1" "$2" "$3" \
    --stages $STAGES --force $STAGES > $LOG/$1_$2_$3.log 2>&1'
for r in $(echo "$RACES" | tr ';' '\n' | grep . | tr ' ' '_'); do
  grep -h ": leader\|Traceback\|Error" $LOG/$r.log
done
