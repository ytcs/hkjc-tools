"""Field/draw/gate-break coupling gains for the generative first-passage pricer.

Relocated here (2026-06, the lean causal refactor) from the deleted ``hkjc.model.constants``
when the acausal additive-regression production model (``fit.py`` / ``features.py`` and the
meta-signal gain tables) was purged. These are the ONLY constants the live causal-physics
pricer (:mod:`hkjc.model.generative`, :mod:`hkjc.model.coupling`) still reads.

PROVISIONAL — known mechanistic debt, NOT first-principles yet. Each of these is a fitted
coupling that the causal program intends to REPLACE with mechanism, not keep:

  * ``DRAW_*``      — bend ground-loss as fractional extra effective distance. Should emerge
                      from track geometry (curvature × lane offset), not a per-venue gain.
  * ``DRAW_VAR_*`` / ``FIGVAR_*`` — draw / figure-spread → diffusion ``D`` (trip variance).
                      Should emerge from the diffusion model and the EVT field-size corollary,
                      not a redistribution gain.
  * ``GATE_BREAK_*`` — start-phase drift/variance reference. Should emerge from the EOM
                      opening transient, not a sprint-reference length.

They are carried verbatim (same env-var overrides, same defaults — most ship OFF/0.0) so the
pricer is byte-stable across the refactor. Replacing them with mechanism is tracked research,
not a constant to re-tune. See ``docs/PHILOSOPHY.md`` (first-principles-not-regression) and
``docs/CAUSAL_MODEL.md``.
"""
from __future__ import annotations

import os

# --- DRAW: bend ground-loss as fractional extra effective distance (the drift) -------------
# DRAW_GAIN: fractional extra effective distance per draw off the rail (bend ground-loss).
DRAW_GAIN = float(os.environ.get("DRAW_GAIN", "0.003"))
# Per-venue bend ground-loss (HV bend-dominated ⇒ larger). ST straight-1000m chute is draw-zeroed.
DRAW_GAIN_BY_VENUE = {"ST": 0.003, "HV": 0.008}
# AWT (dirt) rail bias; resolves by SURFACE, overriding the venue dict.
DRAW_GAIN_AWT = float(os.environ.get("DRAW_GAIN_AWT", "0.008"))
# DRAW_FS_GAIN: crowding×draw — field-size scaling of bend ground-loss; interim stopgap, HV-only.
DRAW_FS_GAIN = float(os.environ.get("DRAW_FS_GAIN", "0.0"))
DRAW_FS_GAIN_BY_VENUE = {"ST": 0.0, "HV": 0.25}

# --- DRAW_VAR / FIGVAR: per-horse diffusion D (trip variance, not the drift) ----------------
# DRAW_VAR_GAIN: draw → diffusion D; per-horse D_i = D·(1+gain·risk_i). ST-only.
DRAW_VAR_GAIN = float(os.environ.get("DRAW_VAR_GAIN", "0.0"))
DRAW_VAR_GAIN_BY_VENUE = {"ST": 0.5, "HV": 0.0}
# FIGVAR_GAIN: per-horse figure-spread → diffusion D. Mean-normalised redistribution (conserves
# the population-mean D). NO-OP at gain=0; ships OFF.
FIGVAR_GAIN = float(os.environ.get("FIGVAR_GAIN", "0.0"))
FIGVAR_GAIN_BY_VENUE = {"ST": 0.0, "HV": 0.0}

# --- GATE_BREAK: start-phase drift/variance reference --------------------------------------
GATE_BREAK_GAIN = float(os.environ.get("GATE_BREAK_GAIN", "0.0"))
GATE_BREAK_GAIN_BY_VENUE = {"ST": 0.0, "HV": 0.0}
GATE_BREAK_FS_GAIN = float(os.environ.get("GATE_BREAK_FS_GAIN", "0.0"))
GATE_BREAK_D_GAIN = float(os.environ.get("GATE_BREAK_D_GAIN", "0.0"))
GATE_BREAK_LREF = float(os.environ.get("GATE_BREAK_LREF", "1200.0"))  # sprint reference trip (m)
