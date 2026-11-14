"""Threshold sensitivity: where does (ball_quiet AND act_quiet) sustain in alive vs D regions?"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from work.analyze import load_all  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402

W = 51  # 2 s rolling window


def main():
    feats = load_all()
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    runs = pm.valid_runs(gt)
    segs = pm.segments(gt, runs)
    vis = feats["ball_vis"] > 0.5
    bs = feats["ball_speed"]

    bs_roll = pd.Series(np.where(vis, bs, np.nan)).rolling(W, center=True, min_periods=5).median()
    act = (feats["mean_speed_0"] + feats["mean_speed_1"]) / 2.0
    act_roll = pd.Series(act).rolling(W, center=True, min_periods=5).mean()
    bs_roll = bs_roll.to_numpy()
    act_roll = act_roll.to_numpy()

    def sustain_runs(cond):
        out = []
        i = 0
        n = len(cond)
        while i < n:
            if cond[i]:
                j = i
                while j < n and cond[j]:
                    j += 1
                out.append((i, j))
                i = j
            else:
                i += 1
        return out

    for ball_th in (0.5, 1.0, 1.5):
        for act_th in (1.0, 1.25, 1.5):
            ball_quiet = ~vis | ((bs_roll < ball_th))
            act_quiet = act_roll < act_th
            cond = ball_quiet & act_quiet
            runs_ev = sustain_runs(cond)
            alive_hits, dead_hits = [], []
            for s, e in runs_ev:
                # label by overlap with D segs
                in_dead = any(max(s, a) < min(e, b) for a, b, k in segs if k == 2)
                # where inside the window does it fall
                if 11185 <= s < 18685:
                    (dead_hits if in_dead else alive_hits).append((s, e, e - s))
            long_alive = [x for x in alive_hits if x[2] >= 50]
            dead_covered = [x for x in dead_hits if x[2] >= 50]
            print("ball<{:.1f} act<{:.2f}: alive-false-runs>=50f: {:d} (lens {}) | dead-runs>=50f: {:d} (starts {})".format(
                ball_th, act_th, len(long_alive),
                sorted({x[2] for x in long_alive})[:8],
                len(dead_covered),
                sorted({x[0] for x in dead_covered})[:8]))


if __name__ == "__main__":
    main()
