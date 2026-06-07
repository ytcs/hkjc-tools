"""Finishing-time LAW: is the stochastic part of T diffusive (IG/Wald) or jump/incident?

DESCRIPTIVE diagnostic (NOT a predictive feature → no as-of/leak-free constraint). Touches
no production code; reuses the par/band logic of features.py (par = within-band mean of the
per-runner finishing time) but reimplemented locally so this study stays self-contained and
read-only on src/.

THE QUESTION (THEORY §2.8, §3e; journal #33 / 0029):
  The production win model (SIM=wald) treats a horse's finishing time T as the first-passage
  of a drift-diffusion to the race length → T ~ Inverse-Gaussian (Wald), mean μ=L/v, shape
  λ=L²/2D, giving CV(T)∝1/√L. #33 confirmed that moment law on turf and found a real
  distance-INDEPENDENT variance FLOOR on AWT (σ0≈0.46s). This study tests the SHAPE/TAIL
  behind those moments:

  - DIFFUSIVE world : T|band well-fit by Inverse-Gaussian; standardized residuals' upper tail
    matches IG's own mild positive skew; no excess upper-tail mass.
  - JUMP/INCIDENT world : troubled trips (checked, blocked, slow break) make a horse SLOWER,
    never faster → an asymmetric HEAVY UPPER tail, roughly distance-independent. The empirical
    upper tail is heavier than IG predicts; an IG-core + low-probability "trouble" add-on
    (shifted exponential) fits the tail materially better; AWT's #33 floor is jump mass, not D.

METHOD
  1. Per-runner T within tight (surface, venue, distance, going) bands. ε = T − par(band).
  2. Fit by MLE within each band: (a) Inverse-Gaussian/Wald [the model's law], (b) Birnbaum-
     Saunders (fatiguelife, a fatigue first-passage with fatter tail), (c) log-normal,
     (d) Gamma, (e) a 2-component mixture IG-core + shifted-exponential "trouble" (explicit
     jump). Report AIC/BIC.
  3. TAIL calibration: empirical-vs-fitted coverage at the 90/95/99th pctiles; upper-tail QQ;
     fitted trouble rate p_trouble and magnitude (mean extra seconds).
  4. Asymmetry: upper- vs lower-tail spread of standardized ε (the sharp incident signature).
  5. AWT floor resolution: is AWT excess variance (vs turf) concentrated in the upper tail
     (→ jump) or spread uniformly (→ larger D)?  The single sharpest discriminator.

Run:  uv run --group bayes python studies/finishing_time_law.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import polars as pl
import scipy.stats as st
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT_DIR = "studies/out"
PANEL = "data_cache/results_rich.parquet"

# tight bands: keep the two large clean goings; surface from Course; AWT only at ST.
KEEP_GOING = ("GOOD", "GOOD TO FIRM")
MIN_BAND_N = 250  # enough runners for a stable 99th-pctile tail


# --------------------------------------------------------------------------- data
def load_bands() -> pl.DataFrame:
    df = pl.read_parquet(PANEL)
    df = df.filter(pl.col("FinishTime") > 0)
    df = df.with_columns(
        pl.when(pl.col("Course").str.contains("ALL WEATHER"))
        .then(pl.lit("AWT"))
        .otherwise(pl.lit("turf"))
        .alias("surface"),
        pl.col("RaceDistance").cast(pl.Int64).alias("L"),
        pl.col("Going").cast(pl.Utf8).alias("going"),
        (
            pl.col("date").cast(pl.Utf8)
            + "_"
            + pl.col("Venue")
            + "_"
            + pl.col("RaceNumber").cast(pl.Utf8)
        ).alias("race_id"),
    )
    df = df.filter(pl.col("going").is_in(KEEP_GOING))
    df = df.with_columns(
        pl.concat_str(
            [pl.col("surface"), pl.col("Venue"), pl.col("L"), pl.col("going")], separator="|"
        ).alias("band")
    )
    # par = within-band mean finishing time (the (going,distance[,surface,venue]) condition
    # standard — the same object features.py builds, here on T not v); residual ε = T − par.
    df = df.with_columns(pl.col("FinishTime").mean().over("band").alias("par_T"))
    df = df.with_columns((pl.col("FinishTime") - pl.col("par_T")).alias("eps"))
    return df


# --------------------------------------------------------------------------- fits
# All families are fit to y = ε − min(ε) + δ (par-relative residual, shifted strictly positive)
# with loc FIXED at 0. This makes every family share the SAME support floor so the comparison is
# on SHAPE/TAIL alone (a free per-family loc would let each distribution game the floor and make
# AIC/tail-coverage incomparable). y inherits ε's mean-zero centring from the band par, so the
# location is the condition standard and the shape is the pure stochastic law we are testing.
SHIFT_DELTA = 0.05  # s; tiny offset so min(y) > 0 (strict positivity for IG/BS/gamma/lognorm)


def to_shifted(eps):
    return eps - eps.min() + SHIFT_DELTA


def fit_invgauss(y):
    p = st.invgauss.fit(y, floc=0.0)
    ll = np.sum(st.invgauss.logpdf(y, *p))
    return ("InvGauss", p, st.invgauss, ll, 2)


def fit_bs(y):
    try:
        p = st.fatiguelife.fit(y, 0.3, floc=0.0)
    except Exception:
        p = st.fatiguelife.fit(y, floc=0.0)
    ll = np.sum(st.fatiguelife.logpdf(y, *p))
    return ("BirnbaumSaunders", p, st.fatiguelife, ll, 2)


def fit_lognorm(y):
    p = st.lognorm.fit(y, floc=0.0)
    ll = np.sum(st.lognorm.logpdf(y, *p))
    return ("LogNormal", p, st.lognorm, ll, 2)


def fit_gamma(y):
    p = st.gamma.fit(y, floc=0.0)
    ll = np.sum(st.gamma.logpdf(y, *p))
    return ("Gamma", p, st.gamma, ll, 2)


# ------- explicit JUMP model: IG BULK core + low-prob shifted-exponential "trouble" tail
# f(y) = (1-pi)·IG(y; mu, scale)  +  pi·[ IG(·; mu, scale) ⊛ Exp(rate=1/tau) ](y)
# The trouble component is the IG core delayed by a one-sided exponential penalty (mean tau): a
# trouble trip adds a positive, exponentially-distributed time loss on top of the clean run.
#
# IDENTIFIABILITY (the crux). An UNCONSTRAINED mixture degenerates to pi→1 with tiny tau, because
# IG is a mediocre fit to the BULK (gamma/lognorm beat it) so the optimizer abuses the trouble
# convolution to merely WIDEN the core rather than add a discrete heavy tail. We prevent that:
#   (1) the IG core (mu, scale) is fixed by fitting IG to the TRIMMED bulk (lower BULK_Q of y),
#       so it represents the diffusive BODY, not the whole sample;
#   (2) pi is bounded to a plausible trouble RATE pi ∈ (0, PI_MAX];
#   (3) tau is bounded to a MATERIAL incident cost tau ∈ [TAU_MIN, tau_max] (a real check/block
#       costs ≳ 1 length ≈ 0.6–2 s, never a 0.3 s nudge).
# Then a large ΔAIC over the bulk-IG AND tail coverage rising to ~1.0 is a genuine JUMP signal;
# pi pinned near 0 / tau at its floor with negligible gain is DIFFUSIVE.
BULK_Q = 0.92      # IG core fit to the lower 92% of residuals (the diffusive body)
PI_MAX = 0.30      # max plausible per-run trouble rate
TAU_MIN = 1.0      # s; a real incident costs at least ~1 length
TAU_MAX_FRAC = 0.7


def _ig_jump_density(y, mu, scale, pi, tau, grid, expw, dg):
    core = st.invgauss.pdf(y, mu, loc=0.0, scale=scale)
    yy = y[:, None] - grid[None, :]
    ig_shift = np.where(yy > 0, st.invgauss.pdf(np.clip(yy, 1e-9, None), mu, loc=0.0, scale=scale), 0.0)
    trouble = (ig_shift * expw[None, :]).sum(axis=1) * dg
    return (1 - pi) * core + pi * trouble


def fit_bulk_ig(y):
    """IG fit to the lower BULK_Q of y → the diffusive BODY (mu, scale), loc=0."""
    cut = np.quantile(y, BULK_Q)
    yb = y[y <= cut]
    p = st.invgauss.fit(yb, floc=0.0)
    return p[0], p[2]


def fit_ig_jump(y, ig_p):
    mu, scale = fit_bulk_ig(y)  # IG core = the BULK fit (diffusive body), then add the heavy tail
    span = y.max()
    tau_max = max(TAU_MIN * 1.5, TAU_MAX_FRAC * span)
    grid = np.linspace(0.0, span, 96)
    dg = grid[1] - grid[0]

    def negll(theta):
        z_pi, z_tau = theta
        pi = PI_MAX / (1.0 + np.exp(-z_pi))
        tau = TAU_MIN + (tau_max - TAU_MIN) / (1.0 + np.exp(-z_tau))
        expw = np.exp(-grid / tau) / tau
        dens = _ig_jump_density(y, mu, scale, pi, tau, grid, expw, dg)
        dens = np.clip(dens, 1e-300, None)
        return -np.sum(np.log(dens))

    best = None
    for pi0 in (0.02, 0.06, 0.12, 0.22):
        for tau0 in (TAU_MIN * 1.5, max(TAU_MIN * 2, span * 0.2), max(TAU_MIN * 3, span * 0.4)):
            tau0 = min(max(tau0, TAU_MIN * 1.01), tau_max * 0.99)
            z_tau0 = np.log((tau0 - TAU_MIN) / (tau_max - tau0))
            z_pi0 = np.log((pi0 / PI_MAX) / (1 - pi0 / PI_MAX))
            try:
                r = minimize(negll, np.array([z_pi0, z_tau0]), method="Nelder-Mead",
                             options={"maxiter": 2500, "xatol": 1e-4, "fatol": 1e-4})
                if best is None or r.fun < best.fun:
                    best = r
            except Exception:
                continue
    if best is None:
        return ("IG+Jump", None, None, -1e12, 4, {})
    z_pi, z_tau = best.x
    pi = PI_MAX / (1.0 + np.exp(-z_pi))
    tau = TAU_MIN + (tau_max - TAU_MIN) / (1.0 + np.exp(-z_tau))
    ll = -best.fun
    # k=4: 2 core (mu,scale from the bulk) + 2 incident (pi,tau).
    info = {"pi": pi, "tau": tau, "mu": mu, "scale": scale, "loc": 0.0, "span": span}
    return ("IG+Jump", best.x, None, ll, 4, info)


def fit_bulk_ig_full(y):
    """Pure 'bulk IG' model: IG fit to the lower BULK_Q, scored on ALL y. This is the diffusive
    null the jump model adds to — its tail UNDER-coverage (if any) is the room a jump could fill."""
    mu, scale = fit_bulk_ig(y)
    ll = np.sum(st.invgauss.logpdf(y, mu, loc=0.0, scale=scale))
    return ("BulkIG", (mu, 0.0, scale), st.invgauss, ll, 2)


def aic_bic(ll, k, n):
    return 2 * k - 2 * ll, k * np.log(n) - 2 * ll


# --------------------------------------------------------------------------- tail coverage
def coverage_table(x, fits, qs=(0.90, 0.95, 0.99)):
    """For each fitted family: model-predicted upper-tail quantile and the EMPIRICAL exceedance
    fraction above it (well-calibrated → matches 1−q). Reports excess (emp/(1−q))."""
    out = {}
    n = len(x)
    for name, p, dist, ll, k, *rest in fits:
        rows = {}
        for q in qs:
            if name == "IG+Jump":
                info = rest[0]
                qhat = ig_jump_ppf(q, info, x)
            else:
                qhat = dist.ppf(q, *p)
            emp = float(np.mean(x > qhat))
            rows[q] = (qhat, emp, emp / (1 - q))
        out[name] = rows
    # empirical quantiles for reference
    emp_q = {q: float(np.quantile(x, q)) for q in qs}
    return out, emp_q, n


def ig_jump_cdf(t, info, xref):
    grid = np.linspace(0.0, info["span"], 80)
    dg = grid[1] - grid[0]
    expw = np.exp(-grid / info["tau"]) / info["tau"]
    t = np.atleast_1d(np.asarray(t, dtype=float))
    core = st.invgauss.cdf(t, info["mu"], loc=0.0, scale=info["scale"])
    tt = t[:, None] - grid[None, :]
    ig_shift = np.where(
        tt > 0,
        st.invgauss.cdf(np.clip(tt, 1e-9, None), info["mu"], loc=0.0, scale=info["scale"]),
        0.0,
    )
    trouble = (ig_shift * expw[None, :]).sum(axis=1) * dg
    return (1 - info["pi"]) * core + info["pi"] * trouble


def ig_jump_ppf(q, info, xref):
    lo, hi = 0.0, xref.max() + 8 * info["tau"]
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if ig_jump_cdf(mid, info, xref)[0] < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- asymmetry
def tail_asymmetry(eps):
    """Standardize ε (median/MAD, robust) and compare upper vs lower tail spread.
    Diffusive IG has MILD positive skew; genuine incidents → excess UPPER mass."""
    med = np.median(eps)
    mad = np.median(np.abs(eps - med)) * 1.4826
    z = (eps - med) / mad
    skew = float(st.skew(eps))
    # tail ratio: 99th-from-median over |1st-from-median|
    up = np.quantile(z, 0.99) - 0.0
    lo = 0.0 - np.quantile(z, 0.01)
    p975 = np.quantile(z, 0.975)
    p025 = -np.quantile(z, 0.025)
    return {
        "skew": skew,
        "tail_ratio_99_01": float(up / lo) if lo > 0 else np.nan,
        "tail_ratio_975_025": float(p975 / p025) if p025 > 0 else np.nan,
        "z99": float(np.quantile(z, 0.99)),
        "z01": float(np.quantile(z, 0.01)),
        "frac_above_3mad": float(np.mean(z > 3)),
        "frac_below_3mad": float(np.mean(z < -3)),
    }


# --------------------------------------------------------------------------- main
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_bands()

    band_n = df.group_by("band").len().filter(pl.col("len") >= MIN_BAND_N).sort("len", descending=True)
    bands = band_n["band"].to_list()
    print(f"Loaded {df.height} runners across {df.select('band').n_unique()} raw bands; "
          f"{len(bands)} bands with n>={MIN_BAND_N}")

    results = []
    for band in bands:
        sub = df.filter(pl.col("band") == band)
        eps = sub["eps"].to_numpy().astype(float)
        y = to_shifted(eps)  # par-relative residual shifted strictly positive; loc fixed at 0
        n = len(y)
        fits = []
        ig = fit_invgauss(y)
        fits.append(ig)
        fits.append(fit_bs(y))
        fits.append(fit_lognorm(y))
        fits.append(fit_gamma(y))
        fits.append(fit_ig_jump(y, ig[1]))
        bulk = fit_bulk_ig_full(y)  # the diffusive null the jump adds to (not in the AIC race)
        cov, emp_q, _ = coverage_table(y, fits + [bulk])
        asym = tail_asymmetry(eps)
        surf, ven, L, going = band.split("|")
        jump_info = fits[-1][5] if len(fits[-1]) > 5 else {}
        results.append(dict(band=band, surface=surf, venue=ven, L=int(L), going=going,
                            n=n, fits=fits, cov=cov, emp_q=emp_q, asym=asym, jump=jump_info, y=y))

    # ---- print AIC/BIC table ----
    print("\n" + "=" * 118)
    print("AIC (lower=better). ΔAIC vs best in band. Win column = lowest-AIC family.")
    print("=" * 118)
    print(f"{'band':30s} {'n':>5s} | " + "".join(f"{nm:>13s}" for nm in
          ["InvGauss", "Birnb-S", "LogNorm", "Gamma", "IG+Jump"]) + "  WINNER")
    fam_wins = {}
    for r in results:
        aics = {}
        for name, p, dist, ll, k, *rest in r["fits"]:
            a, b = aic_bic(ll, k, r["n"])
            aics[name] = a
        best = min(aics, key=aics.get)
        fam_wins[best] = fam_wins.get(best, 0) + 1
        line = f"{r['band']:30s} {r['n']:5d} | "
        for name in ["InvGauss", "BirnbaumSaunders", "LogNormal", "Gamma", "IG+Jump"]:
            d = aics[name] - aics[best]
            line += f"{d:13.1f}"
        line += f"  {best}"
        print(line)
    print(f"\nAIC winners: {fam_wins}")

    # ---- tail coverage table (THE verdict weight) ----
    print("\n" + "=" * 118)
    print("UPPER-TAIL COVERAGE: empirical exceedance / nominal at fitted q-quantile (1.0 = perfect;")
    print(">1 = model UNDER-covers the tail = empirical tail HEAVIER; <1 = OVER-covers).")
    print("BulkIG = IG fit to lower 92%% (diffusive null) — its p99>1 is the room a jump could fill.")
    print("Full-sample InvGauss, Gamma (best core fit) and the IG+Jump shown alongside.")
    print("=" * 118)
    cov_fams = [("BulkIG", "Blk"), ("InvGauss", "IGf"), ("Gamma", "Gam"), ("IG+Jump", "Jmp")]
    hdr = f"{'band':30s} {'n':>5s} |"
    for _, ab in cov_fams:
        for q in (90, 95, 99):
            hdr += f" {ab}{q:>2d}"
    print(hdr)
    for r in results:
        line = f"{r['band']:30s} {r['n']:5d} |"
        for fam, _ in cov_fams:
            for q in (0.90, 0.95, 0.99):
                ratio = r["cov"][fam][q][2]
                line += f" {ratio:5.2f}"
        print(line)
    # summary: median p99 coverage ratio per family across bands
    print("-" * 70)
    for fam, _ in cov_fams:
        r99 = np.array([r["cov"][fam][0.99][2] for r in results])
        r95 = np.array([r["cov"][fam][0.95][2] for r in results])
        print(f"  {fam:10s}  median p95 ratio={np.median(r95):.2f}  "
              f"median p99 ratio={np.median(r99):.2f}  (1.0=calibrated tail)")

    # ---- jump fit summary ----
    print("\n" + "=" * 118)
    print("FITTED TROUBLE COMPONENT (IG+Jump = bulk-IG core + bounded heavy add-on):")
    print("  pi = fitted trouble probability (capped 0.30), tau = mean incident seconds (floor 1.0).")
    print("  pi hitting 0.30 / tau hitting 1.0 = the bound binds → the data WANTS more heavy mass.")
    print("  pi→0 = no jump needed (diffusive). tau ~constant across L = distance-INDEPENDENT incident.")
    print("=" * 118)
    print(f"{'band':30s} {'n':>5s}   {'pi':>6s} {'tau(s)':>7s} {'tau/sd':>7s}")
    pis_all, taus_all, surfs = [], [], []
    for r in results:
        j = r["jump"]
        if not j:
            continue
        sd_eps = float(r["y"].std())
        pis_all.append(j["pi"])
        taus_all.append(j["tau"])
        surfs.append(r["surface"])
        print(f"{r['band']:30s} {r['n']:5d}   {j['pi']:6.3f} {j['tau']:7.2f} "
              f"{j['tau'] / sd_eps:7.2f}")
    pis_all = np.array(pis_all)
    taus_all = np.array(taus_all)
    surfs = np.array(surfs)
    print("-" * 70)
    print(f"  ALL bands : median pi={np.median(pis_all):.3f}  median tau={np.median(taus_all):.2f}s")
    for s in ["turf", "AWT"]:
        m = surfs == s
        if m.sum():
            print(f"  {s:5s}     : median pi={np.median(pis_all[m]):.3f}  "
                  f"median tau={np.median(taus_all[m]):.2f}s  (n_bands={m.sum()})")

    # ---- asymmetry ----
    print("\n" + "=" * 118)
    print("TAIL ASYMMETRY of ε (robust-standardized).  skew>0 & tail_ratio>1 → heavier UPPER tail.")
    print("IG core has MILD positive skew; large excess = incident signature.")
    print("=" * 118)
    print(f"{'band':30s} {'n':>5s} {'skew':>7s} {'TR99/01':>8s} {'z99':>6s} {'z01':>7s} "
          f"{'>3mad%':>7s} {'<3mad%':>7s}")
    for r in results:
        a = r["asym"]
        print(f"{r['band']:30s} {r['n']:5d} {a['skew']:7.2f} {a['tail_ratio_99_01']:8.2f} "
              f"{a['z99']:6.2f} {a['z01']:7.2f} {100*a['frac_above_3mad']:7.2f} "
              f"{100*a['frac_below_3mad']:7.2f}")

    # ---- turf vs AWT tail resolution ----
    print("\n" + "=" * 118)
    print("AWT FLOOR RESOLUTION — is AWT excess variance UNIFORM (→larger D) or UPPER-TAIL (→jump)?")
    print("Compare matched ST distance bands turf vs AWT: lower-tail spread vs upper-tail spread.")
    print("=" * 118)
    awt_turf_compare(df)

    make_plots(results, df)
    print("\nDONE. Plots in studies/out/: finishing_time_densities.png, ft_uppertail_qq.png, "
          "ft_turf_vs_awt_tail.png, ft_trouble_rate.png")


def awt_turf_compare(df: pl.DataFrame):
    """For each ST distance present on BOTH surfaces (turf & AWT, GOOD), compare the standardized
    ε distribution's lower half-spread vs upper half-spread. If AWT's excess is uniform, both
    halves inflate equally (→ larger D); if it is upper-tail, only the upper half inflates (→ jump)."""
    st_df = df.filter((pl.col("Venue") == "ST") & (pl.col("going") == "GOOD"))
    print(f"{'L':>5s} {'surf':>5s} {'n':>5s} {'sd_eps':>7s} {'loSpread':>9s} {'upSpread':>9s} "
          f"{'up/lo':>6s} {'skew':>6s} {'p99-p50':>8s} {'p50-p01':>8s}")
    for L in sorted(st_df["L"].unique().to_list()):
        for surf in ["turf", "AWT"]:
            s = st_df.filter((pl.col("L") == L) & (pl.col("surface") == surf))
            if s.height < 150:
                continue
            e = s["eps"].to_numpy().astype(float)
            med = np.median(e)
            lo_spread = med - np.quantile(e, 0.05)   # median to 5th
            up_spread = np.quantile(e, 0.95) - med   # median to 95th
            p99 = np.quantile(e, 0.99) - med
            p01 = med - np.quantile(e, 0.01)
            print(f"{L:5d} {surf:>5s} {s.height:5d} {e.std():7.3f} {lo_spread:9.3f} "
                  f"{up_spread:9.3f} {up_spread/lo_spread:6.2f} {st.skew(e):6.2f} "
                  f"{p99:8.3f} {p01:8.3f}")


def make_plots(results, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # pick 4 representative large bands: ST turf sprint+route, HV turf, ST AWT
    def pick(surf, ven, L):
        cand = [r for r in results if r["surface"] == surf and r["venue"] == ven and r["L"] == L]
        return cand[0] if cand else None
    picks = [pick("turf", "ST", 1200), pick("turf", "ST", 1800),
             pick("turf", "HV", 1200), pick("AWT", "ST", 1200)]
    picks = [p for p in picks if p]

    # All fits are on y = par-relative residual shifted strictly positive (loc=0). Plot in y-space.
    # --- densities overlay ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, r in zip(axes.flat, picks):
        x = r["y"]
        ax.hist(x, bins=50, density=True, alpha=0.35, color="gray", label="empirical")
        xs = np.linspace(1e-3, x.max(), 400)
        for name, col in [("InvGauss", "C0"), ("BirnbaumSaunders", "C1"),
                          ("LogNormal", "C2"), ("Gamma", "C3")]:
            f = [f for f in r["fits"] if f[0] == name][0]
            ax.plot(xs, f[2].pdf(xs, *f[1]), col, lw=1.5, label=name)
        jf = [f for f in r["fits"] if f[0] == "IG+Jump"][0]
        if jf[5]:
            info = jf[5]
            grid = np.linspace(0.0, info["span"], 80)
            dg = grid[1] - grid[0]
            expw = np.exp(-grid / info["tau"]) / info["tau"]
            core = st.invgauss.pdf(xs, info["mu"], loc=0.0, scale=info["scale"])
            tt = xs[:, None] - grid[None, :]
            igs = np.where(tt > 0,
                           st.invgauss.pdf(np.clip(tt, 1e-9, None), info["mu"],
                                           loc=0.0, scale=info["scale"]), 0.0)
            trouble = (igs * expw[None, :]).sum(axis=1) * dg
            ax.plot(xs, (1 - info["pi"]) * core + info["pi"] * trouble, "k-", lw=2,
                    label=f"IG+Jump (pi={info['pi']:.2f}, tau={info['tau']:.1f}s)")
        ax.set_title(f"{r['band']}  n={r['n']}")
        ax.set_xlabel("par-relative finishing residual  y = ε−min(ε)+δ  (s)")
        ax.legend(fontsize=7)
    fig.suptitle("Fitted finishing-residual densities by band (overlaid families)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/finishing_time_densities.png", dpi=110)
    plt.close(fig)

    # --- upper-tail QQ (IG) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, r in zip(axes.flat, picks):
        x = np.sort(r["y"])
        n = len(x)
        pp = (np.arange(1, n + 1) - 0.5) / n
        igf = [f for f in r["fits"] if f[0] == "InvGauss"][0]
        q_ig = igf[2].ppf(pp, *igf[1])
        mask = pp >= 0.80
        ax.plot(q_ig[mask], x[mask], "C0.", ms=3, label="IG")
        bsf = [f for f in r["fits"] if f[0] == "BirnbaumSaunders"][0]
        q_bs = bsf[2].ppf(pp, *bsf[1])
        ax.plot(q_bs[mask], x[mask], "C1.", ms=3, alpha=0.6, label="Birnbaum-S")
        jf = [f for f in r["fits"] if f[0] == "IG+Jump"][0]
        if jf[5]:
            q_jp = np.array([ig_jump_ppf(p, jf[5], x) for p in pp[mask]])
            ax.plot(q_jp, x[mask], "k.", ms=3, alpha=0.7, label="IG+Jump")
        lo, hi = q_ig[mask].min(), x[mask].max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="ideal")
        ax.set_title(f"{r['band']}  (upper-tail QQ, p>=0.80)")
        ax.set_xlabel("model quantile (s)")
        ax.set_ylabel("empirical quantile (s)")
        ax.legend(fontsize=8)
    fig.suptitle("Upper-tail QQ: points ABOVE the diagonal = empirical tail HEAVIER than model",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/ft_uppertail_qq.png", dpi=110)
    plt.close(fig)

    # --- turf vs AWT tail (standardized eps survival in the upper tail) ---
    fig, ax = plt.subplots(figsize=(9, 6.5))
    st_df = df.filter((pl.col("Venue") == "ST") & (pl.col("going") == "GOOD"))
    for L, style in [(1200, "-"), (1800, "--")]:
        for surf, col in [("turf", "C0"), ("AWT", "C2")]:
            s = st_df.filter((pl.col("L") == L) & (pl.col("surface") == surf))
            if s.height < 150:
                continue
            e = s["eps"].to_numpy().astype(float)
            med = np.median(e)
            mad = np.median(np.abs(e - med)) * 1.4826
            z = np.sort((e - med) / mad)
            surv = 1.0 - (np.arange(1, len(z) + 1) - 0.5) / len(z)
            ax.semilogy(z, surv, col + style, lw=1.6, label=f"{surf} {L}m n={s.height}")
    zz = np.linspace(0, 5, 100)
    ax.semilogy(zz, 1 - st.norm.cdf(zz), "k:", lw=1, label="N(0,1) ref")
    ax.set_xlim(0, 5)
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel("standardized ε  (robust z, upper tail)")
    ax.set_ylabel("survival  P(Z > z)  [log]")
    ax.set_title("Turf vs AWT UPPER-tail (ST GOOD). AWT lifted ONLY in tail → jump; lifted everywhere → larger D")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/ft_turf_vs_awt_tail.png", dpi=110)
    plt.close(fig)

    # --- fitted trouble rate by band ---
    fig, ax = plt.subplots(figsize=(11, 6))
    xs_lbl, pis, taus, cols = [], [], [], []
    cmap = {"turf": "C0", "AWT": "C2"}
    for r in sorted(results, key=lambda r: (r["surface"], r["venue"], r["L"])):
        if not r["jump"]:
            continue
        xs_lbl.append(f"{r['surface'][:1]}{r['venue']}{r['L']}")
        pis.append(r["jump"]["pi"])
        taus.append(r["jump"]["tau"])
        cols.append(cmap[r["surface"]])
    xi = np.arange(len(xs_lbl))
    ax.bar(xi, pis, color=cols, alpha=0.8)
    for i, t in enumerate(taus):
        ax.text(i, pis[i] + 0.002, f"{t:.1f}s", ha="center", fontsize=7)
    ax.set_xticks(xi)
    ax.set_xticklabels(xs_lbl, rotation=90, fontsize=7)
    ax.set_ylabel("fitted trouble probability  pi")
    ax.set_title("Fitted IG+Jump trouble rate per band (label = mean trouble seconds tau). "
                 "AWT (green) vs turf (blue)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/ft_trouble_rate.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
