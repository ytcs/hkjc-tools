"""HKJC data interfaces — everything we read or download from HKJC.

- :mod:`~hkjc.data.race`        one race's facts bundled (`race_info`) — the reconstruction input
- :mod:`~hkjc.data.video`       the four replay MP4 feeds (URLs + cached download)
- :mod:`~hkjc.data.silks`       official racing-silk images
- :mod:`~hkjc.data.tracks`      track geometry (venue x surface x course widths, bends)
- :mod:`~hkjc.data.historical`  official results page (placings, times, running positions)
- :mod:`~hkjc.data.sectional`   per-horse sectional times + positions

Other HKJC sources (racecards, live odds/pools, trackwork, barrier trials, dividends, horse form): fetch recipes in
docs/HKJC_DATA.md and docs/HKJC_SPECS.md.
"""
from .race import RaceInfo, Runner, race_info
from .tracks import TrackGeometry, track_for
from .video import FEEDS, download_race, download_replay, replay_path, replay_url

__all__ = ["RaceInfo", "Runner", "race_info", "TrackGeometry", "track_for", "FEEDS",
           "download_race", "download_replay", "replay_path", "replay_url"]
