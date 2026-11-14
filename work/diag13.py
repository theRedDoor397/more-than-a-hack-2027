"""Aligned broadcast vote vs tactical vote, per GT segment and problem zones."""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import require, source_frames  # noqa: E402
from work.load_atd import load_atd  # noqa: E402
from work.features import team_vote_from_players  # noqa: E402
from work.model import interpolate_ball, rolling, VOTE_SMOOTH  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402

OFFSET = 4010
L, W = 105.3, 68.0
KEEP_DIST = 6.0  # m: aligned ball positions must agree this well


def team_slots(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def main():
    t_ball, t_players = load_atd(require(["atd_meta", "atd_raw"])[1],
                                 source_frames("tactical"), 46)
    b_ball, b_players = load_atd(require(["atd_meta", "atd_raw"], source="broadcast")[1],
                                 source_frames("broadcast"), 51)
    t_team = team_slots(require(["atd_meta", "atd_raw"])[0])
    b_team = team_slots(require(["atd_meta", "atd_raw"], source="broadcast")[0])

    t_int = interpolate_ball(t_ball)
    b_int = interpolate_ball(b_ball)
    _, _, t_logr = team_vote_from_players(t_int, t_players, t_team, sigma=2.5)
    _, _, b_logr = team_vote_from_players(b_int, b_players, b_team, sigma=2.5)

    n_t = len(t_logr)
    bvote = np.full(n_t, np.nan)
    btx = b_ball[:, 0] * L
    bty = b_ball[:, 1] * W
    ttx = t_ball[:, 0] * L
    tty = t_ball[:, 1] * W
    tvis = np.isfinite(ttx)
    for b in range(len(b_logr)):
        t = b + OFFSET
        if not 0 <= t < n_t:
            continue
        if not (np.isfinite(b_logr[b]) and tvis[t]):
            continue
        d = np.hypot(btx[b] - ttx[t], bty[b] - tty[t])
        if d < KEEP_DIST:
            bvote[t] = b_logr[b]

    t_s = rolling(t_logr, 2 * VOTE_SMOOTH + 1, "mean", min_periods=1)
    b_s = rolling(bvote, 2 * VOTE_SMOOTH + 1, "mean", min_periods=1)

    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    segs = pm.segments(gt, pm.valid_runs(gt))

    print("seg  label  len   tacAgree  bcastAgree  bcastCover")
    for i, (a, b, k) in enumerate(segs):
        if k >= 2:
            continue
        sl = slice(a, b)
        ok = np.isfinite(t_s[sl])
        okb = np.isfinite(b_s[sl])
        t_agree = np.mean((t_s[sl][ok] > 0) == (k == 0)) if ok.any() else np.nan
        b_agree = np.mean((b_s[sl][okb] > 0) == (k == 0)) if okb.any() else np.nan
        print("seg {:2d} {}  {:5d}   {:8.3f}    {:8.3f}      {:7.3f}".format(
            i, pm.LABELS[k], b - a, t_agree, b_agree, okb.mean()))

    print("\nproblem zones: tactical vs broadcast vote per 2 s")
    for name, a, b in (("inverted 11351-11624", 11351, 11624),
                       ("rapid 15346-15989", 15346, 15989),
                       ("excursion 15657-15983", 15657, 15983),
                       ("gt16 17293-17380", 17293, 17380)):
        print("== {} ==".format(name))
        for s0 in range(a, b, 50):
            sl = slice(s0, min(s0 + 50, b))
            tv = t_s[sl][np.isfinite(t_s[sl])]
            bv = b_s[sl][np.isfinite(b_s[sl])]
            print("   {:6d}: tac {:7.2f} (n={:2d})  bcast {:7.2f} (n={:2d})".format(
                s0, np.mean(tv) if len(tv) else np.nan, len(tv),
                np.mean(bv) if len(bv) else np.nan, len(bv)))


if __name__ == "__main__":
    main()
