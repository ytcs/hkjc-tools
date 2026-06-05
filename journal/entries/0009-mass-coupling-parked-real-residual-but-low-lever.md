---
title: Mass coupling PARKED -- real residual but low-leverage (b=1 isometric optimal)
id: 0009-mass-coupling-parked-real-residual-but-low-lever
author: modelling
audience: ["all"]
channel: research
kind: deadend
status: confirmed
thread: 
tags: ["mass", "allometric", "parsimony", "low-leverage", "negative-result"]
refs: {}
created: 2026-06-05T16:13:54+00:00
updated: 2026-06-05T16:13:54+00:00
---
Calibrated the steady-state allometric law v_c*(m/m_field)^(b-1) on the quick backtest. b=1.0 (no mass effect) is optimal; b=1.5->47%, b=2->-9%, monotone worse. Fine check: b=1.05/1.10 also worse than 1.0. So despite the confirmed-real bodywt residual (t=+8.8, survives rating+class), coupling mass to v_c does NOT improve goodness-of-fit -- it is LOW-LEVERAGE on the winner's prob, the same lesson as draw. Physical reading: b~1 = ISOMETRIC (aerobic power and carrying-cost both scale ~linearly with mass, cancelling) => no net v_c effect. Per PARSIMONY (physics-plausible + real-residual is NOT sufficient; it must move the needle), mass is left OUT. The high-leverage lever remains W' (+1.4pp). EOM/ALLOMETRIC_B/dynamics.win_probs kept as inert recorded infrastructure (default b=1). Next: speed-figure form (velocity vs rank).
