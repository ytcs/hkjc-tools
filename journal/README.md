# Research Journal

Durable, indexed memory for the agent driving this system — high-value findings
that should survive across sessions. Managed by [`hkjc.ledger`](../src/hkjc/ledger).

- **`entries/*.md`** — the findings (Markdown + frontmatter). The source of truth.
  Append-only in spirit; beliefs are *superseded* or *refuted*, not deleted, so the
  trail of what was thought and when is preserved.
- **`index.jsonl`** — rebuildable search cache (one line per entry). Regenerate with
  `Ledger.reindex()`; never hand-edit.
- **`INDEX.md`** — generated, browsable table of contents grouped by kind.

Usage:

```python
from hkjc.ledger import record, search, recent
record("title", "body…", kind="finding", tags=["clv"])  # kinds: finding,
#   hypothesis, result, decision, gotcha, dataset, deadend
search("closing line")   # ranked list[Finding]
recent(5)
```

Point elsewhere with `$HKJC_JOURNAL` (defaults to this `journal/` dir).
