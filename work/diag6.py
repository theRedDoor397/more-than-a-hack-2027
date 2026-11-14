"""Per-second: ball speed, nearest-player distance d1, activity -- D seg14 vs alive seg20/seg15."""
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
    d1 = np.minimum(feats["d1_0"], feats["d1_1"])
    act = (feats["mean_speed_0"] + feats["mean_speed_1"]) / 2.0

    for name, a, b in (("D seg14", 15889, 16400), ("alive seg20 (H)", 18389, 18685),
                       ("alive seg15 gap area (A)", 16400, 17300)):
        print("{}  [{},{})".format(name, a, b))
        for s0 in range(a, b, 25):
            sl = slice(s0, min(s0 + 25, b))
            v = vis[sl]
            row = "   rel {:4d}  vis {:2d}".format(s0 - a, int(v.sum()))
            if v.any():
                row += "  bs {:5.2f}".format(np.nanmean(bs[sl][v]))
            row += "  d1 {:5.2f}".format(np.nanmedian(d1[sl]))
            row += "  act {:5.2f}".format(np.nanmean(act[sl]))
            print(row)


if __name__ == "__main__":
    main()
