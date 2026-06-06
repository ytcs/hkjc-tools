# Drone-replay path-extraction prototype (studies/cv)

Single-race proof that per-horse `(t, x_long, y_lat)` can be recovered from the HKJC
aerial **drone** replay via classical CV. Disposable scratch — does NOT touch the model.

## Race used
- **HV (Happy Valley), 2026-06-03, Race 1** — "Mount Butler Handicap"
- **Class 5, 1650 M, Turf, "C" Course, Going GOOD TO FIRM**, 12 runners
- Winning time 1:38.85 (avg 16.7 m/s); section splits 27.65 / 23.30 / 23.75 / 24.15
- Course "C": home straight 334 m, **usable width 22.5 m** (the y-bound + the metric
  scale anchor); HV is right-handed, circumference 1450 m (`src/hkjc/core/track.py`).
- Ground-truth running positions (from the result page) for validation:
  winner #3 FAMILY FORTUNE ran 7-7-6-1; #12 led 1-1-1-2; #6 TELECOM POWER 11-11-10-3.

## Video source + how to fetch
- File used: `data_cache/video/drone_20260603_01.mp4` (1920×1080, 25 fps, 143 s),
  already present from prior recon. It is a **multi-angle replay**: aerial **drone**
  for t≈0–110 s (start straight → bends → run to finish), then a ground finish camera.
- URL construction (HKJC video player; see `docs/HKJC_DATA.md` §"Video / replay URLs"):
  ```
  https://racing.hkjc.com/contentAsset/videoplayer_v4/video-player-iframe_v4.html
      ?type=replay-aerial&date=20260603&no=01&lang=eng
  ```
  Media resolves to Akamai `streaminghkjc-a.akamaihd.net/hdflash/...` `_2500kbps.mp4`
  / `.m3u8`. The exact manifest is built client-side in `JCPlayer.js` — capture the
  `<video>` src / network request with the chrome-devtools MCP, then
  `yt-dlp "<m3u8>"` or `ffmpeg -i <m3u8> -c copy out.mp4`. (No structured (x,y,t)
  feed exists — confirmed non-interceptable; pixels are the only route.)

## Pipeline (`extract_path.py`)
1. **Ingest** — ffmpeg/opencv frames at 10 fps on the clean start-straight window
   (t = 6–11 s), where the rails are near-straight white lines and the full field is
   top-down and colour-separable.
2. **Camera-motion compensation** — `keyframe_H_to0`: ORB+RANSAC homography of the
   **static background** (masking out turf + pack), registering every frame to a
   common ground frame via a **keyframe chain** (direct match to the current
   keyframe, promote a new one when overlap drops) so accumulated drift stays bounded
   (bg-align MAE 0→~50 vs 0→150 for a naive consecutive product).
3. **Rail fit** — per frame, Hough white lines validated as **turf boundaries**, the
   two rails **straddling the pack** chosen; lateral separation evaluated at the
   pack's image row (perspective-correct) and locked to 22.5 m → **~32 px/m**.
4. **Detection** — classical foreground (non-green-on-turf), masked to the lane
   **between the two rails**, distance-transform watershed to split the dense pack.
   (Off-the-shelf YOLO finds ~0 horses here — see `../cv_feasibility_probe.py`.)
5. **Tracking** — gap-tolerant NN with constant-velocity prediction, run in **ground
   coordinates** (detections warped to the common frame) so association links the same
   physical horse despite the camera following the screen-stationary pack.
6. **Projection** — `y_lat` = perpendicular metres to the inner rail (per frame);
   `x_long` = cumulative ground arc-length along the rail direction × locked scale.

## Result (committed to `data_cache/path_sample.parquet`)
3 horse tracks over t=6–11 s; columns `[t, horse_id, x_long, y_lat, u, v]`.
Sanity (all pass): **x_long monotonic** (mono=1.00); **y_lat ∈ [0, 15.6] m**, bounded
by the 22.5 m width; **speed 7.5–9.7 m/s** — physical for the opening 5 s off a
standing start (mid-race cruise is ~16.7 m/s).

Outputs in `out/`: `track_xy.png` (x(t),y(t)), `detections.png`, `overlay.png`,
`track_montage.png`.

## Run
```
uv run --group vision --group dev python studies/cv/extract_path.py
```
