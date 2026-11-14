"""Debug long D runs in the 2nd half: tags, ball visibility, restart events."""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import require, source_frames  # noqa: E402
from work.load_atd import load_atd  # noqa: E402
from work.features import extract  # noqa: E402
from work.model import load_tags, rolling, KICK_WIN  # noqa: E402


def main():
    with open(os.path.join(ROOT, "evaluation", "results", "model_v1.csv")) as f:
        rows = list(csv.reader(f))
    labels = np.array([r[1] for r in rows[1:]], dtype="<U1")
    tags = load_tags()

    ball, players = load_atd(require(["atd_meta", "atd_raw"])[1],
                             source_frames("tactical"))
    vis = np.isfinite(ball[:, 0])
    bs = np.zeros(len(ball))
    bs[1:-1] = np.hypot(np.diff(ball[:, 0], 2), np.diff(ball[:, 1], 2)) * 12.5
    bs[~vis] = np.nan
    bs_fast = rolling(np.where(vis, bs, np.nan), KICK_WIN, "median", 3)

    # long D runs in the 2nd half
    runs = []
    prev, start = labels[110808], 110808
    for f in range(110809, 190198):
        if labels[f] != prev:
            runs.append((prev, start, f))
            prev, start = labels[f], f
    runs.append((prev, start, 190198))
    for lab, a, b in runs:
        if lab == "D" and b - a > 200:
            print("D [{:6d},{:6d}) {:5.0f}s".format(a, b, (b - a) / 25))
            cov = [t for t in tags if t["end"] > a and t["start"] < b]
            for t in cov:
                print("    tag [{:6d},{:6d}) {:28s} {:16s} side {}".format(
                    t["start"], t["end"], t["code"], t["type"], t["side"]))
            seg_vis = vis[a:b]
            print("    ball vis in run: {:.2f}".format(seg_vis.mean()))
            # ball visibility per 20 s
            for s0 in range(a, b, 500):
                print("      {:6d}: vis {:.2f}".format(s0, vis[s0:s0 + 500].mean()))


if __name__ == "__main__":
    main()
