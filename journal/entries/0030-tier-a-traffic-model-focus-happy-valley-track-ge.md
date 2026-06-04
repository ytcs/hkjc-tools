---
id: 0030-tier-a-traffic-model-focus-happy-valley-track-ge
title: Tier-A traffic model: focus Happy Valley; track geometry + movable rail captured
kind: decision
status: open
tags: ["traffic-model", "tier-a", "happy-valley", "track-geometry", "movable-rail", "draw-bias"]
refs: {"module": "hkjc.model.track", "source": "racing.hkjc.com/en-us/local/page/racing-course"}
related: []
created: 2026-06-04T07:50:00+00:00
updated: 2026-06-04T07:50:00+00:00
---
User directive: build the Tier-A interacting/traffic-flow race model for ONE racecourse, using real HKJC track specs, and account for the MOVABLE RAIL (the rail shifts out meeting-to-meeting, changing geometry/draw-bias per race -- captured via the per-race `Course` field). Chosen venue: HAPPY VALLEY (strongest draw bias in HK, simplest single 1450m bend-dominated circuit). Specs captured in hkjc.model.track.HAPPY_VALLEY (official figures): circumference 1450m, right-handed/clockwise, course->width A 30.5m ... C+3 19.5m (rail out -> ~11m narrower), home straight ~300-338m. Bend map from sampling points: home straight [0,300], bends [300,770] (Sanatorium/Crescent/The Rock), back straight [770,920], bends [920,1450] (HKFC/Stable+top) -> bend_fraction(1200)~0.63, so draw exposure is persistent. The draw-bias MECHANISM: on a turn, inner lane = shorter arc; low draw holding the rail runs less ground (off-rail metres x turn angle = ground lost). Sha Tin specs also recorded (430m straight, less bias) for later. NEXT: build the cellular-automaton/car-following RaceModel using this geometry + the (v_c,W') energetics engine; draw->initial lane; blocked if horse ahead + no lateral gap. Tier-B (true lateral lanes) needs aerial-video trajectories.
