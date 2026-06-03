"""Research journal — durable, indexed memory for the agent driving this system.

Persist high-value findings across sessions and look them up fast. Entries are
Markdown (the source of truth, under ``$HKJC_JOURNAL`` / ``<repo>/journal``);
``index.jsonl`` is a rebuildable search cache.

See :mod:`hkjc.ledger.store` for details. Quick start::

    from hkjc.ledger import record, search, recent
    record("title", "body...", kind="finding", tags=["clv"])
    search("clv")
"""
from .store import (
    Finding,
    Ledger,
    KINDS,
    STATUSES,
    default_ledger,
    record,
    search,
    recent,
)

__all__ = [
    "Finding", "Ledger", "KINDS", "STATUSES",
    "default_ledger", "record", "search", "recent",
]
