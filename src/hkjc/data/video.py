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
) -> Path:
    """Download one feed (cached). Returns the local path."""
    out = replay_path(date, race_no, feed, root)
    if out.exists() and out.stat().st_size > 0 and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.part")  # per process: concurrent fetches of one file cannot collide
    with requests.get(
        replay_url(date, race_no, feed),
        stream=True,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, out)
    return out


def download_race(date: str, race_no: int, feeds=FEEDS, **kw) -> dict:
    return {f: download_replay(date, race_no, f, **kw) for f in feeds}
