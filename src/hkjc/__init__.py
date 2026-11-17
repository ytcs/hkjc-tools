"""hkjc — HKJC race reconstruction.

Two packages:
- :mod:`hkjc.data`   what the reconstruction reads from HKJC (race facts, replays, silks, track geometry)
- :mod:`hkjc.recon`  race info + the four replay videos -> reconstructed per-horse paths

Subpackages are imported on demand (``import hkjc.data``), never eagerly here.
"""
__version__ = "2.0.0"
