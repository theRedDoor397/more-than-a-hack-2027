"""Check the extended D rule: act<1.5 AND (ball_quiet OR d1>6 OR out-of-pitch) sustained >=2s."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from work.analyze import load_all  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402

W = 51


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
    d1 = np.minimum(feats["d1_0"], feats["d1_1"])

    bs_roll = pd.Series(np.where(vis, bs, np.nan)).rolling(W, center=True, min_periods=5).median()
    act = (feats["mean_speed_0"] + feats["mean_speed_1"]) / 2.0
    act_roll = pd.Series(act).rolling(W, center=True, min_periods=5).mean()
    bs_roll = bs_roll.to_numpy()
    act_roll = act_roll.to_numpy()

    out_pitch = (bx < -1.0) | (bx > 106.3) | (by < -1.0) | (by > 69.0)
    for d1_th in (5.0, 6.0, 7.0, 8.0):
        ball_quiet = ~vis | (bs_roll < 1.5)
        unacc = vis & (d1 > d1_th)
        cond = (act_roll < 1.5) & (ball_quiet | unacc | out_pitch)
        runs_ev = []
        i = 0
        n = len(cond)
        while i < n:
            if cond[i]:
                j = i
                while j < n and cond[j]:
                    j += 1
                runs_ev.append((i, j))
                i = j
            else:
                i += 1
        alive_hits, dead_hits = [], []
        for s, e in runs_ev:
            in_dead = any(max(s, a) < min(e, b) for a, b, k in segs if k == 2)
            if 11185 <= s < 18685:
                (dead_hits if in_dead else alive_hits).append((s, e, e - s))
        long_alive = [x for x in alive_hits if x[2] >= 50]
        dead_covered = [x for x in dead_hits if x[2] >= 50]
        print("d1>{:.0f}: alive-false>=50f: {:d} {} | dead>=50f: {:d} starts {}".format(
            d1_th, len(long_alive), sorted({x[2] for x in long_alive})[:8],
            len(dead_covered), sorted({x[0] for x in dead_covered})))


if __name__ == "__main__":
    main()
