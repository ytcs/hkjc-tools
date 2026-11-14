#!/bin/sh
# Reconstruct the 10-race validation set (venue x surface x course x distance x going x day/night).
# FORCE (env) = comma-separated stages to re-run, e.g. FORCE=detect,hud,rails,observe,identify,fuse,validate
cd "$(dirname "$0")/.." || exit 1
C=scripts/capped.sh
$C uv run --group vision python -m hkjc.recon \
  20260603 HV 8   20260603 HV 3   20260520 HV 2   20260527 HV 3   20260429 HV 3 \
  20260607 ST 6   20260607 ST 5   20260607 ST 8   20260503 ST 7   20260401 ST 6 \
  ${FORCE:+--force $FORCE}
uv run python -c "from hkjc.recon.validate import report; print(report())" | tee data_cache/recon/validation_table.md
