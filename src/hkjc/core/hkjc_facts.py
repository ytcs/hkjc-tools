"""HKJC domain facts — the single machine-readable source for site quirks and rules.

This module is the hard-won "tribal knowledge" about the HKJC data surface and betting
rules, in one place so a refactor (or a new agent) cannot silently lose it. Prose field
guides live in ``docs/HKJC_SPECS.md`` (tracks/rules/pools) and ``docs/HKJC_DATA.md``
(the data surface); this file is their executable mirror — import the
constant instead of hard-coding the magic value at the call site.

Every entry below is a real, verified gotcha that cost time to discover. Read the
comments before changing anything.
"""
from __future__ import annotations

# --------------------------------------------------------------------- venues
VENUES = ("ST", "HV")  # Sha Tin, Happy Valley. An empty venue string makes the
# server return the *current* meeting — convenient live, dangerous in batch.

# ------------------------------------------------------------- scraping quirks
# The sectional split-times page only renders when the date is DD/MM/YYYY. Any other
# format (YYYY-MM-DD, etc.) silently returns an empty "results released shortly" page
# with no tables — which then AttributeErrors on parse. This is THE sectional bug.
SECTIONAL_DATE_FORMAT = "%d/%m/%Y"

# Results / racecard pages are server-rendered HTML. A non-meeting date returns an error
# page with no result tables (not a 404) — always target a real meeting date.

# Live odds: the GraphQL pool payload carries a server ``lastUpdateTime``. That field is
# the stable dedup key — the same reading polled repeatedly shares one lastUpdateTime, so
# dedup on it, not on capture wall-clock.
LIVE_ODDS_DEDUP_KEY = "lastUpdateTime"
LIVEODDS_ENDPOINT = "https://info.cld.hkjc.com/graphql/base/"

# Speed/energy figures come from the current-meeting ``speedguide`` JSON only (the legacy
# ``.xls`` SpeedPro grid is retired). Numeric fields are null until the pre-meeting publish,
# and there is no historical endpoint — capture live or not at all.

# ------------------------------------------------------------- surface detection
# The Course/Track label "ALL WEATHER TRACK" / "AWT" is the AUTHORITATIVE surface signal.
# Going codes are a weak fallback and must be used carefully: GOOD / GD appear in BOTH the
# turf and AWT going scales, so they must NOT trigger AWT. Only AWT-exclusive going codes
# may be used as a fallback.
AWT_LABELS = ("ALL WEATHER TRACK", "AWT")
AWT_DISTANCE = 1650  # the AWT-only trip at Sha Tin (used as a surface heuristic)


def is_awt(course_or_track_label: str | None, distance: int | None = None) -> bool:
    """True if the surface is the all-weather track. Label is authoritative; distance is a hint."""
    if course_or_track_label:
        up = str(course_or_track_label).upper()
        if any(tag in up for tag in AWT_LABELS):
            return True
    return distance == AWT_DISTANCE


def surface_for(course_or_track_label: str | None, distance: int | None = None) -> str:
    """``"awt"`` or ``"turf"`` from the most reliable available signal."""
    return "awt" if is_awt(course_or_track_label, distance) else "turf"


# ------------------------------------------------------------------ place rule
# Number of places paid as a function of field size (HKJC win/place rules).
def places_paid(field_size: int) -> int:
    """3 paid for 7+ runners, 2 for 4-6, 0 for <=3 (no place pool on tiny fields)."""
    if field_size >= 7:
        return 3
    if field_size >= 4:
        return 2
    return 0


# ----------------------------------------------------------------- pool takeout
# Takeout is NOT uniform across pools — netting EV against the wrong rate is a real trap.
# Standard pools ~17.5%; exotics run materially higher (~19.5-25%). Verify against
# docs/HKJC_SPECS.md for the authoritative per-pool schedule before sizing bets.
TAKEOUT = {
    "WIN": 0.175, "PLA": 0.175, "QIN": 0.175, "QPL": 0.175,
    "FCT": 0.205, "TCE": 0.205, "TRI": 0.250, "FF": 0.250, "QTT": 0.250,
    "DBL": 0.175, "TBL": 0.205, "DT": 0.250, "TT": 0.250, "SixUP": 0.250,
}
DEFAULT_TAKEOUT = 0.175

# Pool codes seen across the site: WIN PLA QIN QPL FCT TCE TRI FF QTT DBL TBL DT TT SixUP.
POOL_CODES = tuple(TAKEOUT.keys())


def takeout_for(pool: str) -> float:
    """Takeout rate for a pool code; falls back to the standard 17.5% if unknown."""
    return TAKEOUT.get(str(pool).upper(), DEFAULT_TAKEOUT)
