"""Print the smoothed vote logit per second vs GT labels around flicker zones."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import require, source_frames  # noqa: E402
from work.load_atd import load_atd  # noqa: E402
from work.features import team_vote_from_players, extract  # noqa: E402
from work.model import interpolate_ball, rolling, VOTE_SIGMA, VOTE_SMOOTH, FLIP_LOGIT  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402


def main():
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    vote_path = os.path.join(cache, "vote.npz")
    data = np.load(vote_path, allow_pickle=False)
    logr = data["logr"].copy()
    logr_s = rolling(logr, 2 * VOTE_SMOOTH + 1, "mean", min_periods=1)

    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))

    for name, a, b in (("kickoff zone", 11185, 11450), ("flicker zone 1", 11900, 12800),
                       ("flicker zone 2", 13600, 15000), ("contested", 15300, 15900)):
        print("\n=== {} [{},{}) ===".format(name, a, b))
        for s0 in range(a, b, 50):
            sl = slice(s0, min(s0 + 50, b))
            v = logr_s[sl]
            v = v[np.isfinite(v)]
            g = gt[s0]
            lab = pm.LABELS[g]
            # majority GT label in this 2 s
            gl = pm.LABELS[np.bincount(gt[sl], minlength=4)[:3].argmax()]
            print("  {:6d}  GT {:2s}  logr {:6.2f}  (p25/50/75 {:.2f}/{:.2f}/{:.2f})  n={:2d}".format(
                s0, lab, np.mean(v) if len(v) else float("nan"),
                np.percentile(v, 25) if len(v) else float("nan"),
                np.median(v) if len(v) else float("nan"),
                np.percentile(v, 75) if len(v) else float("nan"), len(v)))


if __name__ == "__main__":
    main()
