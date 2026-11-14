#!/bin/sh
# Run a command under a hard memory cap (default 4G) so a runaway job cannot OOM the machine.
MEM=${MEM:-4G}
exec systemd-run --user --scope --quiet -p MemoryMax=$MEM -p MemorySwapMax=0 "$@"
