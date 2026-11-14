"""Where is the ball visible inside D segs 5/17, and what marks the restarts?"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from work.analyze import load_all  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402


def main():
    feats = load_all()
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    runs = pm.valid_runs(gt)
    segs = pm.segments(gt, runs)
    vis = feats["ball_vis"] > 0.5
    bs = feats["ball_speed"]
    bx = feats["ball_x"]
    by = feats["ball_y"]

    for i, (a, b, k) in enumerate(segs):
        if k != 2:
            continue
        print("D seg {:2d} [{:6d},{:6d})".format(i, a, b))
        idx = np.flatnonzero(vis[a:b]) + a
        print("  visible frames: {}  ({} of {})".format(
            len(idx), len(idx), b - a))
        for f in idx:
            print("    f {:6d}  rel {:5d}  ball ({:6.2f},{:6.2f})  speed {:6.2f}".format(
                f, f - a, bx[f], by[f], bs[f]))
        # restart: first 50 frames after b
        print("  first visible after end: f={}  speed={:.2f}".format(
            idx2 if (idx2 := np.flatnonzero(vis[b:b + 50]) + b).size else None,
            bs[idx2[0]] if (idx2 := np.flatnonzero(vis[b:b + 50]) + b).size else float("nan")))


if __name__ == "__main__":
    main()
