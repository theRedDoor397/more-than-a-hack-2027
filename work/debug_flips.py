"""At flip moments (real GT handovers vs noise excursions): logr and team distances."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from work.model import rolling, VOTE_SMOOTH  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402


def main():
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    data = np.load(os.path.join(cache, "vote.npz"), allow_pickle=False)
    logr = data["logr"].copy()
    logr_s = rolling(logr, 2 * VOTE_SMOOTH + 1, "mean", min_periods=1)

    feats = np.load(os.path.join(cache, "feats.npz"), allow_pickle=False)
    d1_0 = feats["d1_0"]
    d1_1 = feats["d1_1"]

    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    runs = pm.valid_runs(gt)

    spots = {
        # (frame, kind)
        11351: "GT A->H (undetectable zone)",
        11553: "GT H->A",
        11624: "GT A->H",
        12665: "GT H->A",
        13704: "GT H->A",
        13785: "GT A->H",
        14846: "GT H->A",
        15346: "GT A->H (19f spell)",
        15365: "GT H->A",
        15766: "GT A->H (33f spell)",
        15799: "GT H->A",
        17293: "GT A->H",
        15650: "NOISE +3.8 during GT A",
        15850: "NOISE +10.6 during GT A",
        13650: "NOISE -2.5 during GT H",
        14200: "NOISE -3.8 during GT H",
        12400: "NOISE 2.7 during GT H",
    }
    print("frame  kind                      logr   d1h   d1a  d1_min_team  ball_speed")
    for f, kind in sorted(spots.items()):
        lr = logr_s[f]
        d0 = d1_0[f]
        d1 = d1_1[f]
        bs = feats["ball_speed"][f]
        print("{:6d}  {:24s}  {:6.2f}  {:5.2f} {:5.2f}   {:5.2f}     {:6.2f}".format(
            f, kind, lr, d0, d1, min(d0, d1) if np.isfinite(d0) and np.isfinite(d1) else np.nan, bs))

    # for each GT handover: the vote's max excursion toward the new team in a +-3 s window
    print("\nGT handovers: logr before/at/after (means over 1 s windows)")
    for f, frm, to in pm.transitions(gt, runs):
        if to == 2 or frm == 2:
            continue
        before = np.nanmean(logr_s[f - 25:f])
        at = np.nanmean(logr_s[f:f + 25])
        after = np.nanmean(logr_s[f + 25:f + 50])
        # sign: positive favors H
        want = 1 if to == 0 else -1
        print("  f {:6d} {}->{}  want {:+.0f}  before {:6.2f}  at {:6.2f}  after {:6.2f}  d1_new(2s) {:.2f}".format(
            f, pm.LABELS[frm], pm.LABELS[to], want, before, at, after,
            np.nanmean(d1_0[f - 10:f + 40]) if to == 0 else np.nanmean(d1_1[f - 10:f + 40])))


if __name__ == "__main__":
    main()
