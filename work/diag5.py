"""Raw ball-speed timelines inside the D segments: where do fast excursions sit?"""
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

    for i, (a, b, k) in enumerate(segs):
        if k != 2:
            continue
        print("D seg {:2d} [{:6d},{:6d})  (restart = frame {:d})".format(i, a, b, b))
        # print at 1 s resolution: mean and max ball speed per second, vis count
        for s0 in range(a, b, 25):
            sl = slice(s0, min(s0 + 25, b))
            v = vis[sl]
            if v.any():
                print("   {:6d}+1s: vis {:2d}  bs mean {:5.2f} max {:5.2f}  ball ({:6.2f},{:6.2f})".format(
                    s0 - a, int(v.sum()), np.nanmean(bs[sl][v]), np.nanmax(bs[sl][v]),
                    feats["ball_x"][sl][v][0], feats["ball_y"][sl][v][0]))
            else:
                print("   {:6d}+1s: vis  0".format(s0 - a))


if __name__ == "__main__":
    main()
