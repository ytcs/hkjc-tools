"""Top-level orchestration wiring the stages together (SKELETON).

This is the control flow only — every stage call lands in a NotImplementedError
stub. It documents the intended data flow and is import-safe (no heavy deps at
module load). Run order and recommendation follow docs/trajectory/CV_PIPELINE.md:
ALWAYS try the upstream position feed first; build CV (Mode B first) only if not.
"""
from __future__ import annotations

from .schema import RaceContext, RaceTrajectory


def run(ctx: RaceContext, video_path: str, out_path: str) -> RaceTrajectory:
    """End-to-end (stubbed) trajectory extraction for one race.

    Phase 0 (not here): hunt for the raw (x,y) position feed — if found, skip CV.
    """
    from . import ingest, detect, track, identify, homography, project, export

    # 1. INGEST
    reader = ingest.open_video(video_path, target_fps=None)

    # 2. DETECT (+ Mode B marker detection for calibration refs)
    detector = detect.Detector()
    dets_per_frame = [detector.detect(f) for f in reader]            # STUB

    # 3. TRACK -> locally-consistent tracklets
    tracker = track.Tracker(method="botsort")
    for i, dets in enumerate(dets_per_frame):
        tracker.update(i, dets)                                       # STUB
    tracklets = tracker.tracklets()                                   # STUB

    # 4. IDENTIFY -> track_id -> saddlecloth number (global permutation)
    ocr, silks = identify.SaddleclothOCR(), identify.SilksClassifier()
    votes = [identify.vote_tracklet(t, ocr, silks) for t in tracklets]  # STUB
    id_map = identify.assign_runners(votes, field_size=len(tracklets))   # STUB

    # 5. CALIBRATE / HOMOGRAPHY (Mode B: few; Mode A: per-frame chained)
    homs = homography.calibrate_sequence(ctx, marker_dets_per_frame=dets_per_frame)  # STUB
    hom_by_frame = {h.frame: h for h in homs}

    # 6/7. PROJECT + TRAJECTORY per runner
    centreline = project.Centreline(ctx)
    runners = []
    for tl in tracklets:
        runner_no = id_map.get(tl.track_id)                          # STUB mapping
        traj = project.build_trajectory(
            ctx, runner_no, tl.detections, hom_by_frame, centreline
        )                                                            # STUB
        runners.append(traj)
    for r in runners:
        r.events = project.detect_traffic_events(r, runners)         # STUB

    race = RaceTrajectory(context=ctx, runners=runners)

    # 8/9. EXPORT + QA
    race.qa = export.qa(race)                                        # STUB
    export.to_parquet(race, out_path)                               # STUB
    return race
