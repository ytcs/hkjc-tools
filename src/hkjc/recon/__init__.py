"""Race reconstruction: race info + the four HKJC replay videos -> per-horse paths.

    from hkjc.data import race_info
    from hkjc.recon import RaceRun
    run = RaceRun(race_info("20260603", "HV", 8))
    run.run()                                   # cached stages -> data_cache/recon/<key>/
    paths = run.paths()                         # polars: horse_no, t, s, d, lat, sd_s, sd_lat, n_obs

CLI: ``python -m hkjc.recon 20260603 HV 8 [--force stage,...] [--stages ...]``

Coordinates: ``d`` metres to the finish along the racing rail, ``s`` = distance - d metres run,
``lat`` OUTWARD of the racing (movable) rail in the render's lateral units (``lat_m``: metres where a validated map
exists), ``t`` race-clock seconds (0 = gates open).
Architecture and design: docs/RECON.md.
"""

from .pipeline import STAGES, RaceRun

__all__ = ["RaceRun", "STAGES"]
