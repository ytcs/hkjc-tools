#!/usr/bin/env bash
# exact.sh REF_DIR NEW_DIR: compare the core's outputs of two runs of the same cells (det_labels*, core_*.npz,
# paths*.parquet): reports per file whether arrays are identical.
REF=$1; NEW=$2
cd "$(dirname "$0")/../../.." || exit 1
uv run python - "$REF" "$NEW" <<'EOF'
import sys
from pathlib import Path

import numpy as np
import polars as pl

ref, new = Path(sys.argv[1]), Path(sys.argv[2])
bad = 0
for f in sorted(ref.rglob("*")):
    if f.suffix not in (".npz", ".parquet") or not (f.name.startswith(("det_labels", "core_", "paths"))):
        continue
    g = new / f.relative_to(ref)
    if not g.exists():
        print("missing", g)
        bad += 1
        continue
    if f.suffix == ".npz":
        a, b = np.load(f), np.load(g)
        same = set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
    else:
        same = pl.read_parquet(f).equals(pl.read_parquet(g), null_equal=True)
    if not same:
        bad += 1
    print("same" if same else "DIFF", f.relative_to(ref))
print("differ:", bad)
EOF
