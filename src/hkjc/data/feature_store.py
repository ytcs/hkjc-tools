"""Feature store — assemble many ``DataSource``s into one modelling frame.

Joins runner-keyed records from every registered source into a single wide frame
(one row per runner per race), carries the ``date`` column through for
walk-forward splitting, and reports **coverage per source and per language** so a
data gap is measured, not hidden.

Also defines the **Win/Place/Quinella outcome schema** (the labels the first
results-trained model targets, per the scope decision): finishing position, the
field-size-dependent paying-places rule, and the ``won`` / ``placed`` indicators.

Dependency direction: this is data-layer code — it does **not** import the model
or eval layers, and it computes no edge. (The de-vig used in the market feature is
a one-liner kept local on purpose, rather than importing ``eval.clv``.)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import polars as pl

from ..core.contracts import KEY_COLUMNS, RACE_KEYS, CallableSource, DataSource


# --------------------------------------------------------------- outcome schema
def paying_places(field_size: int) -> int:
    """HKJC place rule: 3 paid for 7+ runners, 2 for 4–6, none for ≤3."""
    if field_size >= 7:
        return 3
    if field_size >= 4:
        return 2
    return 0


def add_win_place_outcomes(df: pl.DataFrame, place_col: str = "finish_pos") -> pl.DataFrame:
    """Add the Win/Place outcome columns from a finishing-position column.

    Adds ``field_size`` (runners per race), ``paying_places``, ``won`` (finished
    1st) and ``placed`` (finished within the paying places). Field size is inferred
    per race as the row count — supply complete runner rows per race, or override
    upstream, since dropping non-finishers would understate the field.
    """
    df = df.with_columns(pl.col(place_col).cast(pl.Int64, strict=False).alias("finish_pos"))
    df = df.with_columns(pl.len().over(RACE_KEYS).alias("field_size"))
    places = (
        pl.when(pl.col("field_size") >= 7).then(3)
        .when(pl.col("field_size") >= 4).then(2)
        .otherwise(0)
    )
    df = df.with_columns(places.alias("paying_places"))
    return df.with_columns([
        (pl.col("finish_pos") == 1).cast(pl.Int64).alias("won"),
        ((pl.col("finish_pos") >= 1) & (pl.col("finish_pos") <= pl.col("paying_places")))
        .cast(pl.Int64).alias("placed"),
    ])


# ------------------------------------------------------------- pure adapters
def results_to_outcomes(results: pl.DataFrame) -> pl.DataFrame:
    """Map a ``historical.get_race_data`` frame to the outcome schema.

    Pure transform (no network): selects keys + finishing position from the HKJC
    results table and attaches Win/Place outcomes.
    """
    out = results.select(
        pl.col("Date").cast(pl.Utf8).alias("date"),
        pl.col("Venue").cast(pl.Utf8).alias("venue"),
        pl.col("RaceNumber").cast(pl.Int64, strict=False).alias("race_no"),
        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
        pl.col("Pla").cast(pl.Int64, strict=False).alias("finish_pos"),
        pl.col("WinOdds").cast(pl.Float64, strict=False).alias("result_win_odds"),
    )
    return add_win_place_outcomes(out)


def results_to_features(results: pl.DataFrame) -> pl.DataFrame:
    """Map a ``historical.get_race_data`` frame to runner-keyed model features.

    Pure transform (no network). Extracts what a result page reliably carries:
    the **closing win odds** (de-vigged within the race into ``market_win_prob``
    and its ``log_market_prob`` — the Benter-style backbone feature), the **draw**,
    and **carried/declared weight**. The market feature is computed per race so the
    crowd's probabilities sum to 1; ``log_market_prob`` is the natural linear scale
    for the conditional-logit strength.
    """
    base = results.select(
        pl.col("Date").cast(pl.Utf8).alias("date"),
        pl.col("Venue").cast(pl.Utf8).alias("venue"),
        pl.col("RaceNumber").cast(pl.Int64, strict=False).alias("race_no"),
        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
        pl.col("WinOdds").cast(pl.Float64, strict=False).alias("market_win_odds"),
        pl.col("Dr").cast(pl.Int64, strict=False).alias("draw"),
        pl.col("ActWt").cast(pl.Float64, strict=False).alias("act_wt"),
        pl.col("DeclarHorseWt").cast(pl.Float64, strict=False).alias("decl_wt"),
    )
    inv = pl.when(pl.col("market_win_odds") > 0).then(1.0 / pl.col("market_win_odds")).otherwise(None)
    base = base.with_columns(inv.alias("_inv"))
    base = base.with_columns(
        (pl.col("_inv") / pl.col("_inv").sum().over(RACE_KEYS)).alias("market_win_prob")
    ).drop("_inv")
    return base.with_columns(pl.col("market_win_prob").log().alias("log_market_prob"))


def trailing_form_features(
    panel: pl.DataFrame,
    group_col: str = "horse_id",
    date_col: str = "date",
    finish_col: str = "finish_pos",
    won_col: str = "won",
) -> pl.DataFrame:
    """Per-run recent-form features computed from each horse's **earlier** runs only.

    Leakage-controlled: every feature for a run uses strictly-prior runs of the same
    horse (``shift(1)`` within the horse, ordered by date). Pass a panel containing
    the horse's *full* history (including before the study window) so trailing
    windows aren't truncated. Adds:

    - ``prior_runs`` — career starts before this run
    - ``days_since_last`` — layoff in days
    - ``avg_finish_prior3`` — mean finishing position over the last 3 starts
    - ``win_rate_prior`` — career win rate before this run (null on debut)
    """
    p = panel.sort([group_col, date_col])
    g = group_col
    p = p.with_columns([
        (pl.col(date_col).cum_count().over(g) - 1).alias("prior_runs"),
        (pl.col(date_col) - pl.col(date_col).shift(1).over(g)).dt.total_days().alias("days_since_last"),
        pl.col(finish_col).shift(1).rolling_mean(window_size=3, min_samples=1).over(g).alias("avg_finish_prior3"),
        pl.col(won_col).shift(1).cum_sum().over(g).alias("_wins_prior"),
    ])
    return p.with_columns(
        pl.when(pl.col("prior_runs") > 0)
        .then(pl.col("_wins_prior") / pl.col("prior_runs"))
        .otherwise(None).alias("win_rate_prior")
    ).drop("_wins_prior")


def market_features_from_win(
    win_odds: Sequence[float], date: str, venue: str, race_no: int
) -> pl.DataFrame:
    """Runner-keyed market features from a WIN-odds array (runner i == index i-1).

    Provides ``market_win_odds`` and the de-vigged ``market_win_prob`` — the single
    most important baseline feature (the crowd's signal) and the thing a real model
    must *beat*, not merely reproduce.
    """
    odds = [float(o) if o and o > 0 else None for o in win_odds]
    inv = [1.0 / o if o else 0.0 for o in odds]
    tot = sum(inv)
    probs = [(i / tot) if (tot > 0 and o) else None for i, o in zip(inv, odds)]
    n = len(odds)
    return pl.DataFrame({
        "date": [str(date)] * n,
        "venue": [str(venue)] * n,
        "race_no": [int(race_no)] * n,
        "runner": list(range(1, n + 1)),
        "market_win_odds": odds,
        "market_win_prob": probs,
    })


# ----------------------------------------------------------------- the store
class FeatureStore:
    """Register ``DataSource``s and assemble them into a modelling frame."""

    def __init__(self):
        self.sources: List[DataSource] = []
        self._provenance: Dict[str, str] = {}   # value column -> source name

    def register(self, source: DataSource) -> "FeatureStore":
        self.sources.append(source)
        return self

    # ------------------------------------------------------------- assembly
    def assemble(self, date: str, venue: str) -> pl.DataFrame:
        """Fetch every source for one meeting and join on the runner keys.

        The row universe is the union of runner keys across all sources (so a
        source that only covers some runners doesn't drop the rest). Records which
        column came from which source for coverage reporting.
        """
        fetched: List[Tuple[DataSource, pl.DataFrame]] = []
        for s in self.sources:
            f = s.fetch(date, venue)
            if f is not None and f.height:
                fetched.append((s, f))
        if not fetched:
            return pl.DataFrame()

        universe = pl.concat([f.select(KEY_COLUMNS) for _, f in fetched]).unique(subset=KEY_COLUMNS)
        out = universe
        self._provenance = {}
        for s, f in fetched:
            vals = [c for c in f.columns if c not in KEY_COLUMNS]
            out = out.join(f.select(KEY_COLUMNS + vals), on=KEY_COLUMNS, how="left")
            for c in vals:
                self._provenance[c] = s.name
        return out.sort(KEY_COLUMNS)

    def assemble_meetings(self, meetings: Iterable[Tuple[str, str]]) -> pl.DataFrame:
        """Assemble several ``(date, venue)`` meetings and stack them."""
        frames = [self.assemble(d, v) for d, v in meetings]
        frames = [f for f in frames if f.height]
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    # ------------------------------------------------------------- coverage
    def coverage_report(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Per-source coverage: how many runner-rows each source actually populated.

        ``coverage_pct`` is the share of rows where *any* of the source's provided
        columns is non-null.
        """
        rows = []
        for s in self.sources:
            cols = [c for c in s.provides if c in frame.columns]
            covered = self._covered_count(frame, cols)
            rows.append({
                "source": s.name,
                "role": s.role,
                "languages": ", ".join(s.languages),
                "n_cols": len(cols),
                "rows_total": frame.height,
                "rows_covered": covered,
                "coverage_pct": (covered / frame.height) if frame.height else 0.0,
            })
        return pl.DataFrame(rows)

    def language_coverage(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Coverage rolled up per language — the language-parity instrument.

        A row counts as covered for a language if *any* source of that language
        populated it. Lets a Chinese-vs-English coverage gap surface as a number.
        """
        langs = sorted({lang for s in self.sources for lang in s.languages})
        rows = []
        for lang in langs:
            srcs = [s for s in self.sources if lang in s.languages]
            cols = [c for s in srcs for c in s.provides if c in frame.columns]
            covered = self._covered_count(frame, cols)
            rows.append({
                "language": lang,
                "n_sources": len(srcs),
                "rows_total": frame.height,
                "rows_covered": covered,
                "coverage_pct": (covered / frame.height) if frame.height else 0.0,
            })
        return pl.DataFrame(rows)

    @staticmethod
    def _covered_count(frame: pl.DataFrame, cols: Sequence[str]) -> int:
        cols = [c for c in cols if c in frame.columns]
        if not cols or frame.height == 0:
            return 0
        mask = frame.select(
            pl.any_horizontal([pl.col(c).is_not_null() for c in cols]).alias("m"))["m"]
        return int(mask.sum())


# ------------------------------------------------------------- source builders
_RACECARD_FEATURES = [
    "draw", "rtg", "intl_rtg", "rtg_change", "carried_wt", "decl_horse_wt",
    "wt_change", "best_time", "age", "days_since_last_run", "season_stakes",
    "jockey_allowance", "gear",
]


def racecard_feature_source(races: Sequence[int] = range(1, 12), languages=("en-us",)) -> CallableSource:
    """A `DataSource` of pre-race racecard features (ratings, gear, weight, …).

    Forward-looking (racecards are served only for upcoming meetings). Thin glue
    over :func:`hkjc.data.racecard.get_racecard`; loops the card's races.
    """
    from .racecard import get_racecard

    def fn(date: str, venue: str) -> pl.DataFrame:
        frames = []
        for r in races:
            try:
                rc = get_racecard(date, venue, r)
            except Exception:
                if r == races[0]:
                    return pl.DataFrame()
                break
            if rc.height:
                keep = KEY_COLUMNS + [c for c in _RACECARD_FEATURES if c in rc.columns]
                frames.append(rc.select(keep))
        return pl.concat(frames) if frames else pl.DataFrame()

    return CallableSource(name="racecard", languages=list(languages),
                          provides=list(_RACECARD_FEATURES), fetch_fn=fn, role="feature")


def results_outcome_source(races: Sequence[int] = range(1, 12), languages=("en-us",)) -> CallableSource:
    """A `DataSource` (role=outcome) of Win/Place labels from HKJC results pages.

    Thin glue over :func:`hkjc.data.historical.get_race_data` + the pure
    :func:`results_to_outcomes`; loops the meeting's races until one is missing.
    """
    from .historical import get_race_data

    def fn(date: str, venue: str) -> pl.DataFrame:
        frames = []
        for r in races:
            try:
                frames.append(results_to_outcomes(get_race_data(date, venue, r)))
            except Exception:
                if r == races[0]:
                    return pl.DataFrame()
                break
        return pl.concat(frames) if frames else pl.DataFrame()

    return CallableSource(
        name="results", languages=list(languages),
        provides=["finish_pos", "result_win_odds", "field_size", "paying_places", "won", "placed"],
        fetch_fn=fn, role="outcome")
