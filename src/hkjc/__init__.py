"""hkjc — HKJC race reconstruction.

Two packages:
- :mod:`hkjc.data`   everything read/downloaded from HKJC (results, sectionals, cards, replays, silks)
- :mod:`hkjc.recon`  race info + the four replay videos -> reconstructed per-horse paths

Subpackages are imported on demand (``import hkjc.data``), never eagerly here.
"""
__version__ = "2.0.0"
