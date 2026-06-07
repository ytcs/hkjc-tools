# RESEARCH PROGRAM — the causal, first-principles physics pivot

> The staged roadmap and the concrete, falsifiable research questions (RQs) for the win model's
> rebuild as a **causal, first-principles physics generative model**. This is the *research design*
> document: what to build, in what order, and the pass/fail on each question. **The validation
> hierarchy is inverted from the old model** (PHILOSOPHY §6): the PRIMARY pass/fail is **path-fit +
> cross-surface consistency + re-emergence**; the **market is the LAST yardstick, parked during the
> rebuild** (a paired fixed-seed full-scale check that the causal model surpasses the deleted model's
> recorded benchmark, NOT the fitting target). In the project idiom: **physics leads which
> mechanism to build; consistency + re-emergence gate whether it is real.**
>
> Companion docs (point here, do not restate): `docs/PHILOSOPHY.md` (governing principles + method —
> read first), `docs/THEORY.md` (the physics — the EOM core, identification, the field coupling),
> `docs/CAUSAL_MODEL.md` (the DAG + strip + re-emergence suite), `docs/MODEL_DESIGN.md` (the
> package). The causal model in `hkjc.model` is the ONLY model — the acausal additive model
> (ST 65.0% / HV 59.7% of market) was **DELETED**, not frozen (PHILOSOPHY §9); its number survives as
> a journal-recorded benchmark the causal model must gate *past*.

---

## 0. The framework this program serves (the shared spine)

Three layers, causal + first-principles, leak-free, validated re-emergence-first / market-last:

1. **Latent attribute state** `θ_i = (CP↔v_c, W′_0, P_max, a, …)` — the physical CAUSES, each a
   **physical constant with units and bounds** (PHILOSOPHY §2): aerobic critical velocity `v_c`,
   anaerobic reserve `W′_0`, peak power `P_max`, acceleration/gate `a`, and the future
   control/ground-saved + condition latents. **Hierarchical**: a population prior + a per-horse
   **strictly-prior** posterior (trailing, leak-free).
2. **Generative race model**: `θ_i` + conditions → the **first-principles EOM under bang–CP–bang
   control** (THEORY §III.2, `0047`) + the **mean-field pace/draft coupling** (THEORY §III.4a,
   `0049`), under the diffusion first-passage envelope (`CV(T)∝1/√L`, `0029`) → full predicted
   **path** `(x_long, y_lat, v, t)` per runner → the **joint finishing-order distribution** → every
   pool price. There is **no free coupling gain** — the deployment is the optimal-control solution and
   the field coupling is a physical response (PHILOSOPHY §2, no epicycles).
3. **Identify + validate**:
   - **IDENTIFY by path-reconstruction on the DEDICATED surface, with CROSS-SURFACE CONSISTENCY.**
     Fit each constant by physically-constrained inversion of the laws on the surface where it
     dominates; the same constant measured several ways must AGREE (PHILOSOPHY §4). The dense surface
     resolves what win cannot (`0040`: profile-fit 22–30% better, surfaces `a` at t=+9.98); a
     cross-surface disagreement is a FALSIFICATION naming missing physics (`0048`: CP cruise-vs-Hill
     +1.4 m/s → the field coupling `0049` confirmed).
   - **VALIDATE by RE-EMERGENCE (PRIMARY), the market LAST.** Each stripped meta-signal must
     re-emerge as a predicted consequence (PHILOSOPHY §5; collateral DONE, `0049`). The market
     (%market + ECE on the efficiency-weighted pool book, paired fixed-seed) is the FINAL yardstick,
     **measured to bound the accepted GoF fall, never optimized** (PHILOSOPHY §6).

**The critical lesson — why win is too coarse (`0039`).** Honest path-fit posteriors
*UNDER-discriminate for win*; the honest collapse craters −35/−32pp. An intermediate design read this
as "gate-calibrate the coupling on the market" — **superseded**: `0042`/`0043` showed the win AND
pool-book gates *reject* a strongly path-identified latent (`a`) as a coupling, and the EOM core has
no coupling gain to calibrate. The surviving lesson is the *negative* one: **the win bit cannot
identify mechanism — identify from the dense surface, validate by re-emergence, market last.**

> **READING NOTE FOR §§1–4 (the RQ body).** The detailed RQ templates below were written under the
> intermediate "identify from the surface, then **gate-calibrate the coupling** on the pool book"
> design. **That coupling-calibration step is SUPERSEDED** (PHILOSOPHY §6; §0 above): on the EOM core
> there is no free coupling gain — a constant is *inferred by physically-constrained inversion of the
> laws* and *cross-checked across surfaces* (PHILOSOPHY §2, §4), and the **market is the LAST yardstick,
> not the per-RQ gate**. Read every "*Coupling:* gate-calibrate …" line below as "*Identify:* invert
> the law on the dedicated surface; cross-check consistency; validate by re-emergence; the pool-book
> %market is the final yardstick, measured not optimized." The surface-identification and
> physics-sign content of each RQ stands; the pass/fail is re-emergence-first, market-last.

---

## 1. The staged program

Evolutionary, not revolutionary. The causal model in `hkjc.model` is the ONLY model (the acausal
additive model was deleted, not kept alongside); the gate that confirms a stage is **"does it beat
the deleted model's recorded benchmark on the efficiency-weighted pool book, paired fixed-seed,
full-scale?"**

### Stage 1 — Close the loop on ONE latent: acceleration `a` → start phase, gate-calibrated

The minimal end-to-end proof that the three-layer loop *works*: one identified latent, fit to the
sectional surface, coupled at gate-calibrated strength, validated on win + place.

- **Entry criteria:** the profile-likelihood surface (#40) reproduces; `a` recovered leak-free
  (trailing, `assert_no_lookahead` drift = 0); the `hkjc.model` generative path-sampler can
  emit a start-phase-augmented Wald (`T_i = T_0,i + T_main,i`, §3e Phase 1) and an order
  distribution.
- **Exit criteria (ALL must hold, paired fixed-seed full-scale):**
  1. **WIN gate:** the `a`-augmented model improves **%market AND ECE** at ST (predicted
     strongest cell), at minimum non-negative at HV — the BOTH-improve gate of §5.
  2. **Coupling is gate-calibrated, not honest-posterior:** the coupling strength that maximises
     pool-book %market is *recorded separately* from the honest path-fit posterior strength, and
     the gap is documented (the #40 lesson made concrete — expect honest < gated).
  3. **Distance-band sign matches theory:** the gain lands MOST on ST sprints (§3e: start is a
     large fraction of a short trip; #40 t-gradient t=+10 sprint → +4.8 mile).
  4. **No leak, no regression:** `a` does not degrade any baked latent; the collapse audit (#37)
     confirms `a` is orthogonal to the `v_c` PC1 factor (corr ≈ −0.2, already shown).
- **Deliverable:** `hkjc.model` fits + couples one latent end-to-end and gates ≥ production
  on win. This is the architecture proof.

### Stage 2 — Broaden the surface to margins; gate to the pool BOOK

Extend the fitting surface (velocity profile → + finishing margins) and the gate (win → the
efficiency-weighted pool book). Margins identify reserve/recovery latents the velocity profile
under-resolves; the order pools test the joint distribution win cannot.

- **Entry criteria:** Stage 1 model gates ≥ production on win; a leak-free margins surface is
  joinable (`results_rich.parquet` finish times → within-race margins); the generative model emits
  P(top-k) and pairwise/triple order probabilities; de-vig + efficiency-weighting of the pool book
  is implemented (§4).
- **Exit criteria:**
  1. **W′_0 and ρ identify on the margins surface** with held-out improvement over the
     velocity-only fit (the margins-analogue of #40's profile RMSE win).
  2. **Pool-book gate:** the model improves the **efficiency-weighted pool-book %market** (win +
     place sharp anchors + ≥1 ordering pool [quinella/forecast] + ≥1 set pool [trifecta],
     de-vigged) AND ECE, vs production, paired fixed-seed.
  3. **The order distribution earns its keep:** the model beats a win-marginals-only baseline
     (independent-argmin) on the *ordering* pools specifically — i.e. the joint structure (rank
     correlations / separations) adds measurable pool skill (§4, the exotics RQ).
- **Deliverable:** a pool-book-gated multi-latent generative model, validated on the books win
  cannot see.

### Stage 3 — CV lateral → control latents → geometry × control

Unlock the genuinely new observable — lateral path `y_lat(t)` from CV — and with it the
control/ground-saved latents and the geometry × control game (§3c), the deferred-not-dead axis.

- **Entry criteria:** the #34 CV lateral pipeline clears its binding constraint
  (cross-view re-identification, #41 S1) and produces a leak-free trailing `mean_off / ground_saved`
  trait on a backfilled corpus (start ~100 HV races); bend homography + curvilinear `(d, off)` valid
  through turns (#41 S2/S3).
- **Exit criteria:**
  1. **A control latent identifies on the lateral surface** (ground-saved / path-efficiency)
     distinct from `v_c` — the §3c `O3`/`O4`, now measured from real `y_lat` rather than the
     longitudinal-rank shadow.
  2. **Pool-book gate:** the control latent improves the pool book (expected MOST on HV — bend-
     dominated, strongest HK draw bias) AND ECE, paired fixed-seed; it supersedes (or recalibrates)
     the interim mean-field draw terms (`DRAW_GAIN`, `DRAW_FS_GAIN`, `DRAW_VAR_GAIN`).
  3. **Geometry × control jointly:** draw enters as a *boundary condition on the control policy*
     (§3c), not a scalar shadow — and beats the scalar draw shadow on the pool book.
- **Deliverable:** the geometry × control program with real lateral paths; the 2-D MFG substrate
  for place/exotic pricing (§3c O5). Gait/soundness (#22) is a parallel track, **blocked on footage
  acquisition** (no parade/walk feed in the HKJC URL scheme — #41); do the cheap footage recon (S0)
  but do not invest in the gait model until a walk-up feed is confirmed.

---

## 2. Research questions, per latent

Each RQ follows the same falsifiable template:
**SURFACE SIGNAL that identifies it → how the COUPLING is gate-calibrated → the POOL(S) it should
move + the predicted SIGN/GRADIENT (with the physics cite) → the PASS/FAIL on the pool-book gate.**

### `v_c` — aerobic critical velocity (the ability latent)

- **Status:** identified and live. The #37 factor audit proved the production channel stack is
  **ONE over-measured ability latent** (PC1 ≈ 25%; form + speed-figure + collateral + sectional +
  closing-split are redundant *instruments* of one factor), not multi-attribute.
- **RQ-vc1 (measurement-model collapse).** *Surface:* the multi-section velocity profile (all
  phases) + the trial/gallop instrument channels. *Identify:* fit ONE `v_c` posterior precision-
  weighting every instrument at its OWN precision (races + trials + gallops + closing split), rather
  than a stack of separately-calibrated additive gains that double-count one factor (#37 fix 1).
  *Coupling:* gate-calibrate the fused-posterior → drift coupling (NOT the honest precision — #40).
  *Pool/sign:* must hold win %market AND ECE vs the additive stack (parsimony win, not necessarily a
  %market jump). **PASS/FAIL:** the exact fusion gates ≥ the additive stack on win, paired
  fixed-seed; FAIL if the collapse loses %market (then the additive double-count was load-bearing —
  see #39's "separate gains load-bearing" caution).
- **RQ-vc2 (closing-phase precision channel).** *Surface:* the final-400m sectional (pace + going
  adjusted) — §3e Phase 3, the *least tactically-contaminated* ability readout. *Identify:* enter it
  as its own high-precision instrument in the `v_c` posterior. *Pool/sign:* win + the place pool
  (closing ability sharpens top-k). **PASS/FAIL:** improves pool-book %market + ECE. **Caveat from
  #37:** the raw closing read is 94% spanned by the existing `v_c` (corr 0.95) — it likely converts
  only as a *precision re-weighting*, not a new gain; gate decides.

### `W′_0` — anaerobic reserve

- **Status:** identified as a **distance** reserve `W′ = Σ(v−v_c)^+ t`, drift boost `(1+W′/L)`.
  #35 proved distance-W′ is SUFFICIENT (energy/drag reweighting flat-to-worse both venues — don't
  re-litigate the *recovery shape*). Production carries fade + positional-fade on the W′ knob
  (`FADE_W_GAIN`, `POSFADE_W_GAIN`, §3d T1).
- **RQ-w1 (margins identification).** *Surface:* finishing **margins** (Stage 2) — the reserve
  signature is in *how far* a horse holds or fades over the last sections, which the velocity profile
  under-resolves vs the margin. *Identify:* fit `W′_0` to the margin surface jointly with `v_c`.
  *Coupling:* gate-calibrate the `(1+W′/L)` leverage. *Pool/sign:* the **place** pool primarily
  (reserve is a margin/top-k effect; win-prob impact of W′ is known flat over 0–2 — MODEL_STATE),
  strongest in **sprints** where `W′/L` bites hardest (#35 distance-band test). **PASS/FAIL:** the
  margin-identified `W′_0` improves the place-pool %market + ECE over the velocity-only `W′`.
- **RQ-w2 (intertemporal deployment).** *Surface:* phase-resolved profile — early-burn (front-runner)
  vs late-kick. *Identify:* the kick-fraction / deployment-timing trait (§3e: fade-resistance,
  positional-fade, kick-timing are facets of ONE intertemporal W′-deployment policy). *Coupling:*
  the sign is set by the *phase a trait reads from* — early-committer fades (−), late-committer
  conserves (+). *Pool/sign:* place + forecast (ordering). **PASS/FAIL caveat:** #35 found the
  sit-and-kick deployment *interaction* sub-noise at equal recovered W′ — so this RQ is gated to the
  control layer (a policy trait), not a W′-recovery reweighting; FAIL = re-confirms #35's null.

### `ρ` — recovery (reserve regeneration / mid-race dip recovery)

- **Status:** NOT yet a separate latent. Production recovery is the un-amplified distance form;
  win alone could not resolve a recovery parameter (the #40 batch noted win can't resolve recovery).
- **RQ-rho1 (mid-race-dip surface).** *Surface:* the **mid-trip velocity dip-and-recover** in the
  sectional profile — a horse that surges, dips below `v_c`, then recovers reveals `ρ` (how fast W′
  refills below critical velocity). This is invisible to win and to trip-average speed but explicit
  in the multi-section profile. *Identify:* extend the generative profile to a W′-balance with a
  recovery rate `ρ` (the standard critical-power recovery term) and fit it to the dip shape.
  *Coupling:* gate-calibrate ρ → the closing-phase available reserve. *Pool/sign:* **tactical HV**
  primarily — HV's tight, tactical, mid-race-dominated races (the cruise pace duel, §3b) are where a
  recovery advantage between surges pays; weaker at ST (more even-pace galloping). Place + forecast.
  **PASS/FAIL:** ρ identifies on the dip surface (held-out profile RMSE win in dip-heavy races) AND
  the gate-calibrated coupling improves the HV pool book + ECE. **FAIL = ρ is spanned by `v_c`+`W′_0`
  (the #37 collapse risk — recovery may be a redundant read of the same ability factor); the surface
  identification leg must clear first, exactly the #40 discipline.**

### `a` — acceleration / gate (start phase) — STAGE 1 LATENT

- **Status:** **IDENTIFIED and win-relevant (#40).** `a = v_1/v_cruise` (opening split over settled
  cruise); the 4-latent profile beats the scalar heuristic 22–30% held-out; `a` independent of
  `v_c, W′` (corr ≈ −0.2, a fast opener spends early → lower settled v_c — genuinely orthogonal);
  win signal t(a) = +9.98 ST sprint. NOT yet wired into production.
- **RQ-a1 (the Stage-1 wire-in gate).** *Surface:* the opening-section velocity ramp (#40, the `a`
  fingerprint). *Identify:* trailing leak-free `a` → start-phase drift `T_0,i` (§3e Phase 1:
  `T_i = T_0,i + T_main,i`); its dispersion optionally → `D`. *Coupling:* gate-calibrate the
  start-drift bump — **expect honest < gated (#40)**. *Pool/sign:* **win + place**, sign POSITIVE
  (higher `a` → wins more at equal ability), gradient **strongest ST sprints** (§3e; #40 t-gradient).
  **PASS/FAIL:** improves win %market + ECE at ST, non-negative HV, with the sprint > mile gradient.
  This is the **Stage-1 exit criterion** — the architecture proof.
- **RQ-a2 (gate ability vs gate intent).** *Surface:* the break itself — first-call rank *adjusted
  for draw* (§3e: decompose `early_pos` into draw-explained + gate-skill). *Identify:* split `a` into
  a **capacity** part (raw early acceleration → drift) and a **policy** part (whether the jockey guns
  → a control trait). *Pool/sign:* win + the exotic *forecast* (start order seeds finishing order),
  sharpest where draw bites (§3e: a slow break from a wide gate forfeits longitudinal AND lateral
  ground). **PASS/FAIL caveat:** the *rank-based* gate-break trait already DON'T-BAKED (#30/#37:
  corr 0.998 with `early_pos`, fully spanned). The kinematic `a` converted where the rank trait did
  not — so RQ-a2 must keep the *kinematic opening-velocity* identification, not the rank read; FAIL =
  the policy split is spanned by `a` + the context layer.

### `control / ground-saved` (future, Stage 3)

- **Status:** unbuilt — needs the CV lateral path. The longitudinal-rank shadow (`O3` from
  `trajectory_dataset_full`) is the no-CV proxy; the true `y_lat` is CV-only (#41).
- **RQ-ctl1 (ground-saved index).** *Surface:* the lateral path `y_lat(t)` from CV (#34) — mean
  rail-offset and the arc-length tax (§3c: `∫ χ κ(x) v² y_i dx`, minimised by hugging the rail).
  *Identify:* a per-horse trailing `ground_saved` / path-efficiency trait. *Coupling:* gate-calibrate
  → `v_c` (path-efficiency that saves ground reads as effective speed, §3e). *Pool/sign:* **HV first**
  (bend-dominated, strongest draw bias, smallest fields — #41 backfill choice); win + place.
  **PASS/FAIL:** improves the HV pool book + ECE AND beats the scalar `DRAW_GAIN` shadow (supersedes
  the interim term). **Binding constraint:** cross-view re-id (#41 S1) — without stable named
  identity the path cannot become a leak-free trailing trait.
- **RQ-ctl2 (geometry × control jointly).** *Surface:* draw × the lateral crossing path. *Identify:*
  the control dilemma (commit reserve early to cross to the rail vs stay wide for clear air, §3c) as a
  policy latent with draw as the boundary condition `y_i(0)`. *Pool/sign:* the **exotic** pools
  (quinella/forecast/trifecta) — the 2-D crossing game determines separations, which win marginals
  cannot carry. **PASS/FAIL:** the joint geometry × control structure beats the scalar draw terms on
  the de-vigged ordering pools, paired fixed-seed.

### `condition / soundness` (future, Stage 3 — gated on footage)

- **Status:** **blocked upstream on data** (#41 #22: no parade/walk feed in the HKJC URL scheme;
  in-race footage fails — motion blur, rail-occluded legs). Physiology-from-race-history collapsed to
  ~0 (#MODEL_STATE refuted list — a race outcome is itself a coarse training proxy).
- **RQ-cond1 (debutant soundness, IF footage).** *Surface:* a pre-race parade/walk-up clip → animal
  pose → stride frequency/length + L/R symmetry → a soundness read (#41 S1). *Identify:* a single-clip
  condition latent, leak-free pre-race by construction. *Pool/sign:* win, **specifically debutants /
  lightly-raced** (the one CV signal that helps blank form, the scorecard §5 ✗ cell). **PASS/FAIL:**
  improves debut-cell %market + ECE — BUT **gated entirely on RQ-cond0**: confirm a walk-up feed is
  scrapeable (`type=trackwork` / `btresult` recon). If none exists, **#22 is not actionable — that is
  the finding** (#41). Do not invest in the gait model until S0 succeeds.

---

## 3. Research questions, per surface extension

Each surface extension is an RQ of the form: **what NEW latent does the denser surface unlock, and
what is the BINDING CONSTRAINT to building it?**

### Surface 1 — sectional velocity profile (LIVE, Stage 1)

- **Unlocks:** `a` (acceleration, #40 — proven); the precision re-weighting of `v_c`/`W′` per
  phase (§3e closing-precision channel). The dense surface resolves the start-phase shape win cannot
  (#40: 22–30% better held-out, `a` t=+9.98 invisible to the order statistic).
- **Binding constraint:** none material — `trajectory_dataset_full.parquet` exists, K≥4 sections are
  robust (K=3 noisier — the single-middle-section caveat, #40). This is why Stage 1 starts here.
- **RQ-surf1:** does the profile surface identify ANY further latent beyond `a` and the phase
  re-weightings (e.g. ρ via the mid-race dip, RQ-rho1)? Falsified if the residual profile, after
  `(v_c, W′_0, a)`, is pure per-section measurement noise (a profile-residual PCA flat like the #37
  channel-PCA — one factor + noise).

### Surface 2 — + finishing margins (Stage 2)

- **Unlocks:** `W′_0` and `ρ` at higher resolution (margins carry the reserve/fade signature the
  velocity profile dilutes — RQ-w1, RQ-rho1); and the calibration substrate for the **place** pool
  (margins → top-k separations).
- **Binding constraint:** margins must be joined leak-free (`results_rich.parquet` finish times →
  within-race margins, strictly-prior trailing) — mechanical, low-risk; the discipline is the same
  `assert_no_lookahead` drift = 0 already passing on the profile.
- **RQ-surf2:** do margins identify `W′_0`/`ρ` *beyond* what the velocity profile already pins? FAIL
  if the margin is a deterministic function of the integrated velocity profile (no new information) —
  the held-out margin-RMSE test, analogous to #40's profile-RMSE leg.

### Surface 3 — + lateral path `y_lat(t)` via CV (Stage 3)

- **Unlocks:** the control / ground-saved latents and geometry × control (RQ-ctl1/ctl2) — the only
  source of the `y_i(t)` channel the longitudinal-rank panel structurally cannot give (#41 POC
  resolved rail-saver ≈ 0.8m vs wide ≈ 6.6m end-to-end from pixels).
- **Binding constraint (#41/#34):** **cross-view re-identification.** Homography is solved (#21/#41),
  video is interceptable, compute is modest — the wall is **maintaining stable named identity across a
  full race on a ~20px overhead target** where silks/numbers are sub-resolution. The fix is NOT
  overhead OCR but cross-view fusion (read identity from the broadcast SIDE view → transfer to the
  overhead track by finish-order / time alignment, #41 S1). Plus bend homography (S2) + full-arc
  backfill (S4).
- **RQ-surf3:** does the real `y_lat` ground-saved trait beat the longitudinal-rank `O3` proxy and the
  scalar `DRAW_GAIN` shadow on the pool book? FAIL if the CV lateral noise (the #41 honest caveat: the
  wide track's `ground_lost` is dominated by re-id/detection noise, not real weaving) swamps the
  signal — i.e. re-id quality is the gate, not the mechanism.

---

## 4. Research questions, per validation pool

The gate is the **pool BOOK**, not win alone. Each pool tests a *different piece of model structure*.
The RQ per pool: **what structure does benchmarking it test, and how is it efficiency-weighted /
de-vigged?** All de-vigged (strip the take-out / overround to a true probability), race-normalized,
race-clustered SEs (the residual-diagnosis discipline, PHILOSOPHY §3.3).

- **WIN — P(1st), the sharp anchor.** *Tests:* the win marginal of the order distribution (the order
  statistic's left tail). *Weighting:* highest — sharpest, most efficient pool, the production gate.
  **RQ-pool-win:** does the new latent improve win %market + ECE? The non-negotiable Stage-1 gate.
- **PLACE — P(top-k), the second sharp anchor.** *Tests:* the *top-k mass* of the order distribution —
  the model's calibration in the body of the field, not just the winner. Sensitive to `W′_0`/`ρ`
  (reserve/recovery move *margins* → who fills the minor placings). *Weighting:* high (place is a
  liquid, fairly-efficient HKJC pool). **RQ-pool-place:** does the margins-identified reserve/recovery
  latent improve place %market + ECE *more* than win? (The reserve-is-a-margin-effect prediction —
  MODEL_STATE: W′ win-impact flat but margin-relevant.)
- **QUINELLA — unordered top-2.** *Tests:* the **pairwise co-occurrence** structure of the joint
  distribution (P(both i,j in top-2)) — which independent-argmin marginals get wrong via the rank
  correlations. *Weighting:* medium (an ordering pool, de-vigged; one ordering pool is the §0 gate
  requirement). **RQ-pool-quin:** does the joint generative model beat a win-marginals × independence
  baseline on quinella? This is the cleanest test that the JOINT order distribution carries information
  win cannot.
- **FORECAST — ordered top-2.** *Tests:* the **ordering / separation** between the top two — directly
  sensitive to `a` (start order seeds finishing order, RQ-a1/a2) and to the control crossing game
  (separations, RQ-ctl2). *Weighting:* medium-low (softer than quinella). **RQ-pool-fc:** does adding
  `a` / the control latent improve forecast specifically (the ordered structure), beyond the unordered
  quinella?
- **TRIFECTA — ordered top-3, the set pool.** *Tests:* the **deep joint structure** (third-order
  separations) — the most demanding test of the full order distribution and the 2-D MFG separations
  (§3c O5). *Weighting:* **down-weighted** — the exotic markets are SOFT (less efficient, noisier de-vig,
  larger overround), so a poor trifecta benchmark must not over-penalise a structurally-good model.
  **RQ-pool-tri:** does the model improve trifecta %market — and is the improvement robust to the soft-
  market noise (the RISKS §5 tension)? Treated as confirmatory, not decisive.

**The efficiency weighting (the gate's construction).** The pool-book gate is a *weighted* %market:
**win + place (sharp anchors, full weight) + one ordering pool (quinella/forecast, medium) + one set
pool (trifecta, down-weighted)**, each de-vigged. Exotic softness → down-weight, so the gate's decisive
mass sits on the efficient pools while still *testing* the joint structure on the exotics. **RQ-gate:**
is the efficiency weighting stable run-to-run (paired fixed-seed, `PYTHONHASHSEED=0`)? The MC-seed
±0.3–0.5pp noise band (MODEL_STATE methodology caveat) applies to every pool — softer pools have wider
bands, reinforcing the down-weight.

---

## 5. Risks and open questions (the honest tensions)

1. **Path-fit vs win divergence — the `0039` under-discrimination lesson (the GoF-fall risk).** Honest
   path-fit posteriors *under-discriminate for win*; the honest collapse craters −35/−32pp. The old
   "gate-calibrate the coupling" mitigation is **superseded** (`0042`/`0043`: the gates reject a
   path-identified latent as a coupling; the EOM core has no coupling gain). The current commitment
   (PHILOSOPHY §6) is to **ACCEPT the short-term GoF fall** in exchange for causal confidence, and
   close it by adding causal STRUCTURE (margins, the field equilibrium, CV control) — never by
   re-introducing a stripped meta-signal or by optimizing win% directly. The risk is impatience:
   re-adding a meta-signal to recover win% would undo the rebuild (the ANTI-REGRESSION guardrail).
2. **Identifiability of high-dim `θ` from sparse per-horse data.** The #37 audit is a warning: the
   channel stack collapses to ONE ability factor + noise (PC1 ≈ 25%, 9-of-16 "dimensionality" is
   measurement noise, not attributes). A multi-latent `θ` risks **fitting noise as latents**. Every
   per-latent RQ therefore carries a *surface-identification leg that must clear first* (held-out
   profile/margin RMSE), and a collapse check (is the new latent orthogonal to PC1, or spanned?). ρ and
   the deployment policy are the highest collapse-risk latents.
3. **Compute cost of the state-space fit.** The generative path-sampler + joint order distribution +
   per-horse posterior is far heavier than the production numpy Wald surrogate (~30s full test set).
   **Open question:** can the fit stay tractable (amortised inference? the fast Wald surrogate for the
   win-marginal gate, the full sampler only for exotics — MODEL_STATE: surrogate matches the full sim
   for win, interactions low-leverage)? Risk: the loop becomes too slow to run the paired full-scale
   gate that *is* the gate. **Mitigation:** keep the fast surrogate for win/place; reserve the full
   joint sampler for the order pools (Stage 2+).
4. **Exotic-market softness as a noisy benchmark.** Quinella/forecast/trifecta are less efficient,
   harder to de-vig, larger overround — so they are a *noisy* gate. Risk: a structurally-correct joint
   model is penalised by a soft benchmark, or a lucky overfit passes. **Mitigation:** down-weight
   exotics in the gate (§4); treat trifecta as confirmatory; widen the seed band on soft pools;
   require the *direction* (joint beats independence on quinella, the cleanest ordering test) before
   trusting absolute exotic %market.
5. **The interim draw terms vs geometry × control (deferred-not-dead).** The baked `DRAW_GAIN` /
   `DRAW_FS_GAIN` / `DRAW_VAR_GAIN` are first-order *shadows* of a geometry × control phenomenon
   (MODEL_STATE §A). Risk: Stage-3 control latents must *supersede* these, not stack on them (double-
   counting). RQ-ctl1/ctl2 explicitly gate against the scalar shadow.
6. **CV re-id noise as the real Stage-3 gate.** The #41 POC is honest that the lateral `ground_lost`
   is currently dominated by re-id/detection noise, not real weaving. Risk: Stage 3's binding constraint
   is a *data-quality* wall (re-id), and the mechanism is fine but unmeasurable until it clears.

---

## 6. Prioritized first-wave RQ list (Stage 1 — build/test FIRST)

In order. Each is gate-anchored (pass/fail on the paired fixed-seed pool-book gate) and theory-led.

1. **RQ-a1 — wire `a` into the start-phase drift, gate on WIN.** (THE Stage-1 proof.) Trailing leak-free
   `a` → `T_0,i` (§3e Phase 1); gate-calibrate the coupling (record honest vs gated, #40); PASS =
   win %market + ECE improve at ST with the sprint > mile gradient (#40 t=+10 → +4.8), non-negative HV.
   *This closes the loop on one latent and proves the architecture.*
2. **RQ-vc1 — the measurement-model collapse for `v_c`.** Fuse the redundant ability instruments into
   ONE precision-weighted posterior (#37 fix 1); gate-calibrate the coupling; PASS = win %market + ECE
   ≥ the additive stack (parsimony win); watch the #39 "separate gains load-bearing" failure mode.
   *Establishes the honest measurement model the whole state-space sits on.*
3. **RQ-a2 — split `a` into gate-ability vs gate-intent.** Keep the *kinematic* opening-velocity
   identification (the rank-based gate trait already died, #30/#37); PASS = the policy split adds
   forecast/win %market beyond `a`-capacity alone, not spanned by the context layer.
4. **RQ-vc2 — the closing-phase precision channel.** Final-400m sectional as its own high-precision
   `v_c` instrument (§3e Phase 3); PASS = pool-book %market + ECE improve; FAIL-expected-mode = it is
   the 95%-spanned read (#37) and converts only as a re-weight.
5. **RQ-surf1 — profile-residual identifiability probe.** After `(v_c, W′_0, a)`, is the profile
   residual one factor + noise (#37-style PCA) or does it hide ρ (RQ-rho1)? Decides whether Stage 2's
   margins surface is needed to unlock recovery, or whether the profile already carries it. *Cheap,
   descriptive, sets the Stage-2 entry decision.*

First-wave deferred (Stage 2/3 entry-gated, scoped now, not built first): RQ-w1, RQ-rho1 (need the
margins surface); RQ-ctl1/ctl2, RQ-surf3 (need CV re-id, #41 S1); RQ-cond0/cond1 (blocked on footage).

---

## 7. The discipline (every RQ inherits this — see `docs/PHILOSOPHY.md`)

- **Causal, not correlational; first-principles, not regression.** Build the CAUSES; meta-signals are
  EFFECTS to re-emerge, never inputs. Every parameter is a physical constant with units and bounds —
  no fitted gain / affine map / spline without a physical referent (the no-epicycles rule;
  PHILOSOPHY §1, §2, ANTI-REGRESSION).
- **Leak-free, non-negotiable.** Per-horse latents are STRICTLY-PRIOR trailing; `assert_no_lookahead`
  drift = 0 before any number is trusted (PHILOSOPHY §7).
- **Identify on the dedicated surface, with cross-surface consistency.** Pin each constant where it
  dominates; the same constant measured several ways must AGREE — a disagreement is a falsification
  naming missing physics, never a number to fudge (PHILOSOPHY §4; `0048`/`0049`).
- **Validate re-emergence first; the market LAST.** Re-emergence of the stripped meta-signals is the
  PRIMARY pass/fail; %market + ECE (paired fixed-seed, `PYTHONHASHSEED=0`, full-scale; `quick_eval`
  over-states ~4×) is the FINAL yardstick, measured to bound the accepted GoF fall, never optimized
  (PHILOSOPHY §5, §6). A dead first attempt ≠ a dead axis — but build it as a *law*, not a probe.
- **Evolutionary.** The causal `hkjc.model` is the ONLY model; the acausal model is DELETED, its
  number a journal-recorded benchmark (PHILOSOPHY §9). A stage is confirmed only by gating past that
  benchmark on the efficiency-weighted pool book — the last check, after it earns causal confidence.
