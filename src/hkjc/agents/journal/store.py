"""The cross-session multi-agent journal.

Markdown entries are the source of truth (readable in-context, diff-able, git-durable);
a flat ``index.jsonl`` mirrors metadata + searchable text for fast lookup (rebuildable,
not source of truth). Append-only; status changes are edits so the trail is preserved.

What makes it multi-agent: every entry has an ``author`` and an ``audience``, so
:meth:`Journal.inbox` returns exactly what an agent should read on session start —
entries addressed to it (or to ``all``) that are still ``open``. ``thread`` links replies.
On every write the journal also materialises per-agent ``INBOX-<agent>.md`` and a grouped
``INDEX.md`` so a human (or an agent without the API) can browse the same views.

High-volume operational data (paper-trade fills, live-odds snapshots) does NOT belong
here — it goes to :class:`hkjc.agents.journal.ops_log.OpsLog` (parquet). Only *notable*
ops events get promoted to a markdown ``observation`` entry for another agent.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .schema import ALL, JSON_KEYS, META_KEYS, Entry

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = Path(os.environ.get("HKJC_JOURNAL", str(_REPO_ROOT / "journal")))


def _slug(title: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:maxlen].strip("-") or "entry"


def _render(e: Entry) -> str:
    meta = {k: getattr(e, k) for k in META_KEYS}
    lines = ["---", f"title: {e.title}"]
    for k, v in meta.items():
        if k in JSON_KEYS:
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        else:
            lines.append(f"{k}: {'' if v is None else v}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + e.body.rstrip() + "\n"


def _parse(text: str) -> Entry:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("entry is missing frontmatter")
    meta: Dict[str, object] = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        key, _, val = lines[i].partition(":")
        key, val = key.strip(), val.strip()
        if key in JSON_KEYS:
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                val = [] if key != "refs" else {}
        meta[key] = val
        i += 1
    body = "\n".join(lines[i + 1:]).strip("\n")
    thread = meta.get("thread") or None
    return Entry(
        id=str(meta.get("id", "")), author=str(meta.get("author", "modelling")),
        audience=list(meta.get("audience", [ALL]) or [ALL]), channel=str(meta.get("channel", "research")),
        kind=str(meta.get("kind", "finding")), status=str(meta.get("status", "open")),
        thread=str(thread) if thread else None, tags=list(meta.get("tags", []) or []),
        refs=dict(meta.get("refs", {}) or {}), created=str(meta.get("created", "")),
        updated=str(meta.get("updated", "")), title=str(meta.get("title", "")), body=body)


class Journal:
    """Durable, indexed, multi-agent journal of :class:`Entry` records."""

    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root)
        self.entries_dir = self.root / "entries"
        self.index_path = self.root / "index.jsonl"

    # --------------------------------------------------------------- write API
    def record(
        self,
        title: str,
        body: str = "",
        *,
        author: str = "modelling",
        audience: Sequence[str] = (ALL,),
        channel: str = "research",
        kind: str = "finding",
        status: str = "open",
        thread: Optional[str] = None,
        tags: Sequence[str] = (),
        refs: Optional[Dict[str, object]] = None,
        supersedes: Sequence[str] = (),
        now: Optional[datetime] = None,
    ) -> Entry:
        """Persist a new entry and return it. ``supersedes`` marks prior ids ``superseded``."""
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        e = Entry(id=self._next_id(title), author=author, audience=list(audience) or [ALL],
                  channel=channel, kind=kind, status=status, thread=thread, tags=list(tags),
                  refs=dict(refs or {}), created=ts, updated=ts, title=title, body=body)
        self._write(e)
        for sid in supersedes:
            self.set_status(sid, "superseded", now=now, _reindex=False)
        self.reindex()
        return e

    def reply(self, parent_id: str, title: str, body: str = "", **kw) -> Entry:
        """Record an entry threaded onto ``parent_id`` (same conversation)."""
        kw["thread"] = parent_id
        return self.record(title, body, **kw)

    def update(self, entry_id: str, now: Optional[datetime] = None, _reindex: bool = True, **changes) -> Entry:
        e = self.get(entry_id)
        for k, v in changes.items():
            if not hasattr(e, k):
                raise AttributeError(f"Entry has no field {k!r}")
            setattr(e, k, v)
        e.updated = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        self._write(e)
        if _reindex:
            self.reindex()
        return e

    def set_status(self, entry_id: str, status: str, now: Optional[datetime] = None, _reindex: bool = True) -> Entry:
        return self.update(entry_id, status=status, now=now, _reindex=_reindex)

    # ---------------------------------------------------------------- read API
    def get(self, entry_id: str) -> Entry:
        path = self._path_for(entry_id)
        if path is None:
            raise KeyError(entry_id)
        return _parse(path.read_text(encoding="utf-8"))

    def all(self) -> List[Entry]:
        if not self.entries_dir.exists():
            return []
        return [_parse(p.read_text(encoding="utf-8")) for p in sorted(self.entries_dir.glob("*.md"))]

    def inbox(self, agent: str, *, limit: int = 50) -> List[Entry]:
        """Open entries addressed to ``agent`` (or ``all``), newest first — read on session start."""
        recs = [r for r in self._load_index()
                if r["status"] == "open" and (agent in r["audience"] or ALL in r["audience"])
                and r["author"] != agent]
        recs.sort(key=lambda r: (r["created"], r["id"]), reverse=True)
        return [self.get(r["id"]) for r in recs[:limit]]

    def thread(self, entry_id: str) -> List[Entry]:
        """An entry and everything threaded onto it, oldest first."""
        root = entry_id
        seen = self.get(entry_id)
        root = seen.thread or entry_id
        chain = [e for e in self.all() if e.id == root or e.thread == root]
        return sorted(chain, key=lambda e: (e.created, e.id))

    def search(self, query: str = "", *, author: Optional[str] = None, channel: Optional[str] = None,
               kind: Optional[str] = None, tags: Optional[Sequence[str]] = None,
               status: Optional[str] = None, limit: int = 20) -> List[Entry]:
        """Keyword search over title/tags/body with metadata filters (newest-first on ties)."""
        toks = [t for t in re.split(r"\W+", query.lower()) if t]
        want_tags = set(tags or [])
        scored = []
        for rec in self._load_index():
            if author and rec["author"] != author:
                continue
            if channel and rec["channel"] != channel:
                continue
            if kind and rec["kind"] != kind:
                continue
            if status and rec["status"] != status:
                continue
            if want_tags and not want_tags <= set(rec["tags"]):
                continue
            title, tagstr, text = rec["title"].lower(), " ".join(rec["tags"]).lower(), rec["text"]
            score = sum(3 * (t in title) + 2 * (t in tagstr) + (t in text) for t in toks)
            if toks and score == 0:
                continue
            scored.append((score, rec["created"], rec["id"]))
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return [self.get(i) for _, _, i in scored[:limit]]

    def recent(self, n: int = 10, *, author: Optional[str] = None, channel: Optional[str] = None) -> List[Entry]:
        recs = self._load_index()
        if author:
            recs = [r for r in recs if r["author"] == author]
        if channel:
            recs = [r for r in recs if r["channel"] == channel]
        recs.sort(key=lambda r: (r["created"], r["id"]), reverse=True)
        return [self.get(r["id"]) for r in recs[:n]]

    # ----------------------------------------------------------------- indexing
    def reindex(self) -> int:
        """Rebuild ``index.jsonl``, ``INDEX.md`` and per-agent ``INBOX-<agent>.md``. Returns count."""
        self.root.mkdir(parents=True, exist_ok=True)
        entries = self.all()
        with self.index_path.open("w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(self._index_record(e), ensure_ascii=False) + "\n")
        self._write_index_md(entries)
        self._write_inboxes(entries)
        return len(entries)

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
    def _index_record(e: Entry) -> dict:
        text = " ".join([e.title, " ".join(e.tags), e.body]).lower()
        return {"id": e.id, "title": e.title, "author": e.author, "audience": e.audience,
                "channel": e.channel, "kind": e.kind, "status": e.status, "thread": e.thread,
                "tags": e.tags, "created": e.created, "updated": e.updated,
                "summary": e.summary(), "text": text}

    def _write_index_md(self, entries: List[Entry]) -> None:
        by_author: Dict[str, List[Entry]] = {}
        for e in entries:
            by_author.setdefault(e.author, []).append(e)
        lines = ["# Agent Journal — Index", "",
                 f"{len(entries)} entries. Generated; do not edit by hand (rebuilt by `Journal.reindex`).", ""]
        for author in sorted(by_author):
            lines.append(f"## {author}")
            lines.append("")
            for e in sorted(by_author[author], key=lambda x: x.id):
                to = ",".join(e.audience)
                tags = (" — _" + ", ".join(e.tags) + "_") if e.tags else ""
                lines.append(f"- **{e.id}** →{to} ({e.channel}/{e.status}) "
                             f"[{e.title}](entries/{e.id}.md){tags}")
            lines.append("")
        (self.root / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_inboxes(self, entries: List[Entry]) -> None:
        agents = sorted({a for e in entries for a in (e.audience or [ALL])} - {ALL}
                        | {e.author for e in entries})
        for agent in agents:
            box = [e for e in entries
                   if e.status == "open" and e.addressed_to(agent) and e.author != agent]
            box.sort(key=lambda e: (e.created, e.id), reverse=True)
            lines = [f"# Inbox — {agent}", "",
                     f"{len(box)} open item(s) addressed to {agent}. Read on session start.", ""]
            for e in box:
                lines.append(f"- **{e.id}** from {e.author} ({e.channel}/{e.kind}) "
                             f"[{e.title}](entries/{e.id}.md)")
                lines.append(f"  - {e.summary(160)}")
            (self.root / f"INBOX-{agent}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

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

    def _write(self, e: Entry) -> Path:
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        path = self.entries_dir / f"{e.id}.md"
        path.write_text(_render(e), encoding="utf-8")
        return path

    def _path_for(self, entry_id: str) -> Optional[Path]:
        exact = self.entries_dir / f"{entry_id}.md"
        if exact.exists():
            return exact
        matches = sorted(self.entries_dir.glob(f"{entry_id}*.md")) if self.entries_dir.exists() else []
        return matches[0] if matches else None
