"""Official racing-silk images (per horse brand id), e.g. K057 -> RaceColor/K057.gif (38x30 px)."""

from __future__ import annotations

from pathlib import Path

from ._util import get

_URL = "https://racing.hkjc.com/racing/content/Images/RaceColor/{hid}.gif"
DEFAULT_DIR = Path("data_cache/silks")


def silk_path(horse_id: str, root: Path | str = DEFAULT_DIR) -> Path | None:
    """Download (cached) the silk GIF for a horse brand id; None if unavailable."""
    out = Path(root) / f"{horse_id}.gif"
    if out.exists():
        return out
    r = get(_URL.format(hid=horse_id), headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return out
