"""misses.py CODE_DIR OUT_TSV [KEY ...]: per race of a harness run dir (runs/<tag>) with U and both masked cells done
(lg0/lg1 silk, dc0/dc1 disc): the J of each masked cell's video labelling under U's video evidence minus U's own J
(negative: U's search missed a lower labelling), and the confident-flip counts of P3 (U final prob >= 0.9 / >= 0.99,
label > 0; a flip is a masked final label > 0 that differs). Appends one row per race; races already in OUT_TSV are
skipped."""
import pickle
import sys
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon import core, fusion
from hkjc.recon.pipeline import RaceRun

root, out = Path(sys.argv[1]), Path(sys.argv[2])
keys = sys.argv[3:] or sorted(p.name for p in (root / "U").iterdir())
done = {ln.split("\t")[0] for ln in out.read_text().splitlines()} if out.exists() else set()


def complete(c, k):
    return (root / c / k / "det_labels_final.npz").exists() and (root / c / k / "fuse_video.pkl").exists()


for key in keys:
    if key in done or not complete("U", key):
        continue
    cells = [c for c in ("lg0", "lg1", "dc0", "dc1") if (root / c / key).exists()]
    if len(cells) != 2 or not all(complete(c, key) for c in cells):
        continue
    d, v, n = key.split("_")
    run = RaceRun(race_info(d, v, int(n)), root=root / "U", log=lambda m: None, mask=Mask())
    T = np.load(run.p("tracks.npz"), allow_pickle=True)
    ev = run._evidence(T)
    Lr = core.list_rows(run.legend_rows(), run._tick_of_tracks(), ev.horses)
    V = pickle.load(open(run.p("fuse_video.pkl"), "rb"))
    col = {int(h): i for i, h in enumerate(ev.horses)}
    ex_s, ex_l = run._core_obs(V["fin"], col)
    ts = [T["t"].min(), T["t"].max()] + ([Lr.tau.min()] if len(Lr.tau) else []) + (
        [ex_s.t.min(), ex_s.t.max()] if len(ex_s.t) else [])
    tg = np.arange(min(ts), max(ts) + fusion.DT / 2, fusion.DT)
    m = core.Model(ev, Lr, tg, run.info.width)
    ju = core.evaluate(m, V["seg_lab"], ex_s, ex_l).J
    gaps = []
    for c in cells:
        lab = pickle.load(open(root / c / key / "fuse_video.pkl", "rb"))["seg_lab"]
        gaps.append(core.evaluate(m, lab, ex_s, ex_l).J - ju if len(lab) == ev.n_seg else np.nan)
    zu = np.load(root / "U" / key / "det_labels_final.npz")
    u, p = zu["labels"], zu["prob"]
    row = [key, f"{ju:.1f}"] + [f"{g:.1f}" for g in gaps]
    for thr in (0.9, 0.99):
        conf = (p >= thr) & (u > 0)
        fl = 0
        for c in cells:
            mm = np.load(root / c / key / "det_labels_final.npz")["labels"]
            fl += int((conf & (mm > 0) & (mm != u)).sum())
        row += [str(2 * int(conf.sum())), str(fl)]
    with open(out, "a") as f:
        f.write("\t".join(row) + "\n")
    print("\t".join(row), flush=True)
