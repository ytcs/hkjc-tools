"""Research journal — durable, indexed memory for the agent.

A place to persist *high-value, durable findings* across sessions and look them up
fast. Designed to be ergonomic for the model that drives this system:

- **Entries are Markdown** with a small frontmatter header — readable directly in
  context, diff-able, and git-durable (the entries *are* the source of truth).
- **A flat ``index.jsonl``** mirrors every entry's metadata + searchable text, so
  lookup is a single-file scan, never a directory crawl. It is a rebuildable cache
  (:meth:`Ledger.reindex`), not the source of truth.
- **Append-only** by default; status changes (confirmed / refuted / superseded)
  are edits, so the trail of what was believed and when is preserved.

Typical use::

    from hkjc.ledger import record, search, recent
    record("CLV is the north-star metric",
           "Beating the closing line is the strongest leading indicator ...",
           kind="decision", tags=["clv", "evaluation"])
    search("closing line")          # -> ranked list[Finding]
    recent(5)                       # -> latest entries

Keep it for things worth remembering: confirmed/refuted hypotheses, hard-won
gotchas, design decisions and their rationale, dataset descriptions, dead ends
(so they're not re-explored). Not a scratchpad.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path(os.environ.get("HKJC_JOURNAL", str(_REPO_ROOT / "journal")))

# Loose vocabularies — not enforced, just the conventions the agent should reuse.
KINDS = ("finding", "hypothesis", "result", "decision", "gotcha", "dataset", "deadend")
STATUSES = ("open", "confirmed", "refuted", "superseded", "archived")

_META_KEYS = ("id", "title", "kind", "status", "tags", "refs", "related", "created", "updated")
_JSON_KEYS = ("tags", "refs", "related")


@dataclass
class Finding:
    """One journal entry."""

    id: str
    title: str
    kind: str = "finding"
    status: str = "open"
    tags: List[str] = field(default_factory=list)
    refs: Dict[str, object] = field(default_factory=dict)
    related: List[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    body: str = ""

    def summary(self, n: int = 200) -> str:
        flat = " ".join(self.body.split())
        return flat[:n] + ("…" if len(flat) > n else "")

    def __str__(self) -> str:  # pragma: no cover - display helper
        tags = (" [" + ", ".join(self.tags) + "]") if self.tags else ""
        return f"{self.id} ({self.kind}/{self.status}){tags}: {self.title}"


def _slug(title: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:maxlen].strip("-") or "entry"


def _render(f: Finding) -> str:
    meta = {k: getattr(f, k) for k in _META_KEYS}
    lines = ["---"]
    for k, v in meta.items():
        if k in _JSON_KEYS:
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + f.body.rstrip() + "\n"


def _parse(text: str) -> Finding:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("entry is missing frontmatter")
    meta: Dict[str, object] = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        key, _, val = lines[i].partition(":")
        key, val = key.strip(), val.strip()
        if key in _JSON_KEYS:
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                val = []
        meta[key] = val
        i += 1
    body = "\n".join(lines[i + 1:]).strip("\n")
    return Finding(
        id=str(meta.get("id", "")),
        title=str(meta.get("title", "")),
        kind=str(meta.get("kind", "finding")),
        status=str(meta.get("status", "open")),
        tags=list(meta.get("tags", []) or []),
        refs=dict(meta.get("refs", {}) or {}),
        related=list(meta.get("related", []) or []),
        created=str(meta.get("created", "")),
        updated=str(meta.get("updated", "")),
        body=body,
    )


class Ledger:
    """Durable, indexed store of :class:`Finding` entries."""

    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root)
        self.entries_dir = self.root / "entries"
        self.index_path = self.root / "index.jsonl"

    # --------------------------------------------------------------- write API
    def record(
        self,
        title: str,
        body: str,
        kind: str = "finding",
        tags: Sequence[str] = (),
        status: str = "open",
        refs: Optional[Dict[str, object]] = None,
        related: Sequence[str] = (),
        supersedes: Sequence[str] = (),
        now: Optional[datetime] = None,
    ) -> Finding:
        """Persist a new finding and return it.

        ``supersedes`` marks the listed entry ids as ``superseded`` (and links
        them via ``related``), so the journal records *why* an old belief was
        replaced rather than silently dropping it.
        """
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        related = list(related) + [s for s in supersedes if s not in related]
        f = Finding(
            id=self._next_id(title),
            title=title,
            kind=kind,
            status=status,
            tags=list(tags),
            refs=dict(refs or {}),
            related=related,
            created=ts,
            updated=ts,
            body=body,
        )
        self._write(f)
        for sid in supersedes:
            self.set_status(sid, "superseded", now=now, _reindex=False)
        self.reindex()
        return f

    def update(self, entry_id: str, now: Optional[datetime] = None, _reindex: bool = True, **changes) -> Finding:
        """Edit fields of an existing entry (e.g. ``status``, ``body``, ``tags``)."""
        f = self.get(entry_id)
        for k, v in changes.items():
            if not hasattr(f, k):
                raise AttributeError(f"Finding has no field {k!r}")
            setattr(f, k, v)
        f.updated = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        self._write(f)
        if _reindex:
            self.reindex()
        return f

    def set_status(self, entry_id: str, status: str, now: Optional[datetime] = None, _reindex: bool = True) -> Finding:
        return self.update(entry_id, status=status, now=now, _reindex=_reindex)

    # ---------------------------------------------------------------- read API
    def get(self, entry_id: str) -> Finding:
        path = self._path_for(entry_id)
        if path is None:
            raise KeyError(entry_id)
        return _parse(path.read_text(encoding="utf-8"))

    def all(self) -> List[Finding]:
        if not self.entries_dir.exists():
            return []
        return [_parse(p.read_text(encoding="utf-8")) for p in sorted(self.entries_dir.glob("*.md"))]

    def search(
        self,
        query: str = "",
        *,
        kind: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Finding]:
        """Keyword search over title/tags/body, with optional metadata filters.

        Scoring weights title matches over tags over body. An empty ``query`` with
        filters acts as a filtered listing (most recent first).
        """
        toks = [t for t in re.split(r"\W+", query.lower()) if t]
        want_tags = set(tags or [])
        scored = []
        for rec in self._load_index():
            if kind and rec["kind"] != kind:
                continue
            if status and rec["status"] != status:
                continue
            if want_tags and not want_tags <= set(rec["tags"]):
                continue
            title, tagstr, text = rec["title"].lower(), " ".join(rec["tags"]).lower(), rec["text"]
            score = 0
            for t in toks:
                score += 3 * (t in title) + 2 * (t in tagstr) + (t in text)
            if toks and score == 0:
                continue
            scored.append((score, rec["created"], rec["id"]))
        # score desc, then most-recent first, then id — all descending.
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return [self.get(i) for _, _, i in scored[:limit]]

    def recent(self, n: int = 10, *, kind: Optional[str] = None) -> List[Finding]:
        recs = self._load_index()
        if kind:
            recs = [r for r in recs if r["kind"] == kind]
        recs.sort(key=lambda r: (r["created"], r["id"]), reverse=True)
        return [self.get(r["id"]) for r in recs[:n]]

    def by_tag(self, tag: str, limit: int = 50) -> List[Finding]:
        return self.search("", tags=[tag], limit=limit)

    # ----------------------------------------------------------------- indexing
    def reindex(self) -> int:
        """Rebuild ``index.jsonl`` and ``INDEX.md`` from the entries. Returns count."""
        self.root.mkdir(parents=True, exist_ok=True)
        findings = self.all()
        with self.index_path.open("w", encoding="utf-8") as fh:
            for f in findings:
                fh.write(json.dumps(self._index_record(f), ensure_ascii=False) + "\n")
        self._write_index_md(findings)
        return len(findings)

    def _load_index(self) -> List[dict]:
        if not self.index_path.exists():
            if self.entries_dir.exists():
                self.reindex()
            else:
                return []
        out = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    @staticmethod
    def _index_record(f: Finding) -> dict:
        text = " ".join([f.title, " ".join(f.tags), f.body]).lower()
        return {
            "id": f.id, "title": f.title, "kind": f.kind, "status": f.status,
            "tags": f.tags, "created": f.created, "updated": f.updated,
            "summary": f.summary(), "text": text,
        }

    def _write_index_md(self, findings: List[Finding]) -> None:
        by_kind: Dict[str, List[Finding]] = {}
        for f in findings:
            by_kind.setdefault(f.kind, []).append(f)
        lines = ["# Research Journal — Index", "",
                 f"{len(findings)} entries. Generated; do not edit by hand "
                 "(rebuilt by `Ledger.reindex`).", ""]
        for kind in sorted(by_kind):
            lines.append(f"## {kind}")
            lines.append("")
            for f in sorted(by_kind[kind], key=lambda x: x.id):
                tags = (" — _" + ", ".join(f.tags) + "_") if f.tags else ""
                lines.append(f"- **{f.id}** ({f.status}) [{f.title}](entries/{f.id}.md){tags}")
            lines.append("")
        (self.root / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    # ----------------------------------------------------------------- internal
    def _next_id(self, title: str) -> str:
        nums = []
        if self.entries_dir.exists():
            for p in self.entries_dir.glob("*.md"):
                head = p.stem.split("-", 1)[0]
                if head.isdigit():
                    nums.append(int(head))
        n = (max(nums) + 1) if nums else 1
        return f"{n:04d}-{_slug(title)}"

    def _write(self, f: Finding) -> Path:
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        path = self.entries_dir / f"{f.id}.md"
        path.write_text(_render(f), encoding="utf-8")
        return path

    def _path_for(self, entry_id: str) -> Optional[Path]:
        exact = self.entries_dir / f"{entry_id}.md"
        if exact.exists():
            return exact
        # allow lookup by numeric prefix alone (e.g. "0007")
        matches = sorted(self.entries_dir.glob(f"{entry_id}*.md")) if self.entries_dir.exists() else []
        return matches[0] if matches else None


# ----------------------------------------------------------- module-level sugar
_default: Optional[Ledger] = None


def default_ledger() -> Ledger:
    """The shared journal at ``$HKJC_JOURNAL`` (defaults to ``<repo>/journal``)."""
    global _default
    if _default is None:
        _default = Ledger()
    return _default


def record(*args, **kwargs) -> Finding:
    """Record a finding in the default journal. See :meth:`Ledger.record`."""
    return default_ledger().record(*args, **kwargs)


def search(*args, **kwargs) -> List[Finding]:
    """Search the default journal. See :meth:`Ledger.search`."""
    return default_ledger().search(*args, **kwargs)


def recent(*args, **kwargs) -> List[Finding]:
    """Most-recent entries in the default journal. See :meth:`Ledger.recent`."""
    return default_ledger().recent(*args, **kwargs)
