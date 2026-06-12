"""Cache per-frame rail curves (racing/fixed split) for the horse-tagging pipeline.

Reuses the promoted rail reconstruction (aerial_rail.track_sequence over the LOOP-1 cache)
and stores every frame's curve points keyed ``{frame}_{racing|fixed}`` so the horse detector
can apply the venue-model in-track gate without re-running the rail pipeline.

Run:  PYTHONHASHSEED=0 uv run --group bayes python studies/cv_htag_rails_cache.py R08 R02A
Out:  data_cache/_htag_rails_{race}.npz
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")
from hkjc.vision.feeds.aerial_rail import track_sequence  # noqa: E402


def main():
    for race in sys.argv[1:] or ["R08", "R02A"]:
        zc = np.load(f"data_cache/_railclean_{race}.npz", allow_pickle=True)
        seq = track_sequence(list(zc["keys"]), zc["clean"])
        out = {}
        for f, curves in seq.items():
            for name, flag in (("racing", True), ("fixed", False)):
                pts = [np.asarray(c["pts"], np.float32) for c in curves
                       if bool(c.get("running")) == flag]
                out[f"{f}_{name}"] = np.vstack(pts) if pts else np.zeros((0, 2), np.float32)
        np.savez_compressed(f"data_cache/_htag_rails_{race}.npz", **out)
        print(race, "frames:", len(seq), "-> data_cache/_htag_rails_%s.npz" % race)


if __name__ == "__main__":
    main()
