"""HKJC replay videos — four auth-free deterministic MP4 feeds per race (Akamai).

    aerial  the Aerial Virtual View (split screen: live camera left, synthetic 3D render right)
    drone   real high camera following the field (one continuous shot gate -> finish), shares the
            RF edit timeline
    full    race film (RF): side-on tracking camera, burnt-in race clock + "NNNm TO FINISH"
    patrol  stewards' fixed high mount, head-on on the straights

URL pattern: .../hdflash/replay-{feed}/{YYYY}/{YYYYMMDD}/{NN}/eng/
replay-{feed}_{YYYYMMDD}_{NN}_eng_{kbps}kbps.mp4
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import requests

FEEDS = ("aerial", "drone", "full", "patrol")
_URL = (
    "https://streaminghkjc-a.akamaihd.net/hdflash/replay-{feed}/{y}/{d}/{n:02d}/eng/"
    "replay-{feed}_{d}_{n:02d}_eng_{kbps}kbps.mp4"
)
DEFAULT_DIR = Path("data_cache/video")


def _ymd(date: str) -> str:
    return str(date).replace("-", "").replace("/", "")


def replay_url(date: str, race_no: int, feed: str, kbps: int = 2500) -> str:
    if feed not in FEEDS:
        raise ValueError(f"feed must be one of {FEEDS}")
    d = _ymd(date)
    return _URL.format(feed=feed, y=d[:4], d=d, n=int(race_no), kbps=kbps)


def replay_path(date: str, race_no: int, feed: str, root: Path | str = DEFAULT_DIR) -> Path:
    return Path(root) / f"{feed}_{_ymd(date)}_{int(race_no):02d}.mp4"


def download_replay(
    date: str,
    race_no: int,
    feed: str,
    root: Path | str = DEFAULT_DIR,
    overwrite: bool = False,
    timeout: float = 60.0,
    tries: int = 3,
) -> Path:
    """Download one feed (cached). Returns the local path.

    The body streams into a uniquely named sibling (a random name opened exclusively: unique even between
    hosts sharing the volume, where process ids repeat), renamed onto the cache path only when complete
    (the byte count matches ``Content-Length`` when the server sends one) and removed on any failure, so
    concurrent or interrupted fetches never collide or leave a partial file. Timeouts, dropped
    connections, truncated bodies and 5xx responses are retried (backoff 2, 4 s)."""
    out = replay_path(date, race_no, feed, root)
    if out.exists() and out.stat().st_size > 0 and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        tmp = out.parent / f".{out.stem}.{uuid.uuid4().hex}.part"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        try:
            with os.fdopen(fd, "wb") as f, requests.get(
                replay_url(date, race_no, feed),
                stream=True,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as r:
                if r.status_code >= 500 and k < tries - 1:
                    raise _TransientError(f"HTTP {r.status_code}")
                r.raise_for_status()
                n = 0
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
                want = r.headers.get("Content-Length")
                if want and "Content-Encoding" not in r.headers and n != int(want):
                    raise _TransientError(f"truncated: {n} of {want} bytes")
            os.replace(tmp, out)
            return out
        except (_TransientError, requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
            if k == tries - 1:
                raise
            time.sleep(2.0 * (k + 1))
        finally:
            tmp.unlink(missing_ok=True)
    raise AssertionError("unreachable")


class _TransientError(IOError):
    """A transient download failure (5xx, truncated body)."""


def download_race(date: str, race_no: int, feeds=FEEDS, **kw) -> dict:
    return {f: download_replay(date, race_no, f, **kw) for f in feeds}
