"""Exogenous race conditions — the boundary conditions θ is generated *into*.

A :class:`RaceConditions` bundles everything the generative model needs about a race that is
NOT a per-horse latent: the trip, venue/surface, per-runner draw, field size, track geometry.
These are the §2.5 environment + §2.6 structure terms — known before the race is run (so they
carry no leak), supplied by the data loader. Kept separate from θ so the generative model has a
clean (θ_i, conditions) → law signature and the leak boundary is obvious: θ is strictly-prior
per-horse; conditions are pre-race public facts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RaceConditions:
    """Pre-race public conditions for one race (no outcomes, no latents).

    Attributes
    ----------
    race_id:    stable race identifier (for the deterministic sampler seed).
    distance:   trip in metres (the first-passage barrier L).
    venue:      "ST" / "HV" (per-venue calibration key).
    surface:    "turf" / "awt" (diffusion multiplier key).
    draws:      per-runner barrier draw (shape ``(n,)``), ordered like the θ list.
    field_size: number of runners n.
    band:       coarse distance band ("sprint"/"mile"/"middle"/"staying") for the prior.
    geom:       optional ``hkjc.core.track`` geometry (bend fraction etc.); None in the skeleton.
    section_dists: optional per-section metres for the K-section profile likelihood (#41).
    """

    race_id: str
    distance: int
    venue: str
    surface: str
    draws: np.ndarray
    field_size: int
    band: str = "mile"
    geom: object | None = None
    section_dists: np.ndarray | None = None
    # Per-runner / per-race auxiliary geometry the production wald law reads to assemble the EXACT
    # (μ, λ) — bend ground-loss draw, gate-break effective-L, per-horse D variance knobs. Carried
    # here (pre-race public facts, no leak) so the GenerativeRaceModel reproduces production
    # byte-for-byte at a=OFF (Acceptance Gate A). All optional; absent ⇒ the skeleton's bare law.
    course: str = "A"
    gate_break_shrunk: np.ndarray | None = None   # EB-shrunk, within-race-centred break skill
    fig_disp: np.ndarray | None = None            # per-horse strictly-prior speed-figure spread
    # FIELD/DEPLOYMENT cause (journal 0065): per-runner anaerobic reserve [m] the front-group pace WAR
    # depletes EARLY (``field.pace_war_burn_per_runner`` from the field's strictly-prior style/pace
    # composition — KNOWN pre-race, leak-free). It is a REDUCTION of W′_eff in the generative drift,
    # applied ONLY when GenerativeRaceModel(field_pace_war=True); absent / None ⇒ no field term (the
    # byte-identical baseline). Non-negative, ordered like the θ list.
    pace_war_burn: np.ndarray | None = None
