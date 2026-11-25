"""tvframes.py KEY CELLDIR OUT T...: the TV (full) feed and the aerial frame at race times T, synced on the race clock
(aerial: clock_f0 / 25; full: viz.full_offset, the aerial left panel's lag), saved as one image per time."""
import sys
from pathlib import Path

import av
import cv2
import numpy as np

from hkjc.data import race_info
from hkjc.recon import viz
from hkjc.recon.pipeline import RaceRun

key, celldir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
import os
FEED = os.environ.get("FEED", "full")
ts = [float(x) for x in sys.argv[4:]]
d, v, n = key.split("_")
run = RaceRun(race_info(d, v, int(n)), root=celldir, log=lambda m: None)
a0 = run.clock_f0() / 25.0
f0, lags, cors = viz.full_offset(run)
print("aerial offset", a0, "full offset", f0, "lags", lags, "cors", np.round(cors, 3))


def grab(path, sec):
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        c.seek(int(max(sec - 1, 0) / st.time_base), stream=st)
        for fr in c.decode(st):
            if float(fr.pts * st.time_base) >= sec - 0.02:
                return fr.to_ndarray(format="bgr24")
    return None


out.mkdir(parents=True, exist_ok=True)
for t in ts:
    A = grab(run.video("aerial"), a0 + t)
    F = grab(run.video(FEED), f0 + t)
    h = 720
    A = cv2.resize(A, (int(A.shape[1] * h / A.shape[0]), h))
    F = cv2.resize(F, (int(F.shape[1] * h / F.shape[0]), h))
    img = np.hstack([F, A])
    cv2.putText(img, f"{key} race t={t:.1f} s  TV (left) | aerial (right)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)
    cv2.imwrite(str(out / f"{key}_{FEED}_{t:05.1f}.jpg"), img)
    print("wrote", t)
