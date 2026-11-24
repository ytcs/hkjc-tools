"""s_subobs: the fixed race list. Per set (dev201 sorted, gate149 sorted) and venue: eligible races (not sealed, not
already picked, turf, stored fuse_video.pkl, official_subsplits non-empty), 10 at an even stride from the first.
J_INTERIOR = list index mod (number of boundaries - 1). Run from the worktree root (data_cache symlinked).
-> data_cache/s_subobs/races.txt: SET KEY SRC J_INTERIOR NBND"""
import os

import hkjc
from hkjc.data import race_info
from hkjc.data.race import official_boundaries, official_subsplits

print("hkjc from", hkjc.__file__)
R = "data_cache"
SEAL = set(open(f"{R}/index/circle_sealed.txt").read().split())
SETS = {"dev": sorted(open(f"{R}/index/product201.txt").read().split()),
        "gate": sorted(open(f"{R}/s_g150/gate149.txt").read().split())}
SRC = {"dev": lambda k: f"{R}/product/{k}", "gate": lambda k: f"{R}/s_g150/hb/none/{k}"}
seen, out = set(), []
for s, ks in SETS.items():
    for ven in ("HV", "ST"):
        el = []
        for k in ks:
            if k.split("_")[1] != ven or k in SEAL or k in seen or not os.path.exists(f"{SRC[s](k)}/fuse_video.pkl"):
                continue
            d, v, n = k.split("_")
            info = race_info(d, v, int(n))
            if not official_subsplits(info)[0] or info.surface != "turf":
                continue
            el.append(k)
        step = len(el) / 10
        for i in range(10):
            k = el[int(i * step)]
            assert k not in SEAL
            d, v, n = k.split("_")
            info = race_info(d, v, int(n))
            nb = len(official_boundaries(info)[0])
            out.append(f"{s} {k} {SRC[s](k)} {i % (nb - 1)} {nb}")
            seen.add(k)
        print(s, ven, len(el), "eligible")
open(f"{R}/s_subobs/races.txt", "w").write("\n".join(out) + "\n")
print(len(out))
