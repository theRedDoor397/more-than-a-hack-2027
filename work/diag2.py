"""Diagnostics: weighted-vote agreement, ball-missing gap structure, D evidence."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import GAME, require, source_frames  # noqa: E402
from work.load_atd import load_atd  # noqa: E402
from work.features import extract, nearest_team, team_vote_from_players  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402
from work.analyze import load_all  # noqa: E402


def main():
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    ball, players = load_atd(raw_path, n_frames)
    import xml.etree.ElementTree as ET
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team_of_slot = np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)

    feats = load_all()
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    runs = pm.valid_runs(gt)
    segs = pm.segments(gt, runs)

    nt = nearest_team(feats)
    v0, v1, logr = team_vote_from_players(ball, players, team_of_slot, sigma=2.5)

    print("per-segment agreement: plain nearest vs weighted vote (logr sign)")
    for i, (a, b, k) in enumerate(segs):
        sl = slice(a, b)
        if k >= 2:
            continue
        ok = nt[sl] >= 0
        okv = np.isfinite(logr[sl]) & (v0[sl] + v1[sl] > 1e-6)
        agree_nt = np.mean(nt[sl][ok] == k) if ok.any() else np.nan
        agree_v = np.mean((logr[sl][okv] > 0) == (k == 0)) if okv.any() else np.nan
        print("  seg {:2d} {} len {:5d}: nearest {:.3f}   vote {:.3f}   voteframes {:.3f}".format(
            i, pm.LABELS[k], b - a, agree_nt, agree_v, okv.mean()))

    # ball-missing run structure: alive vs dead
    print("\nball-missing run lengths (frames) in window, alive vs dead")
    vis = feats["ball_vis"] > 0.5

    def gap_hist(sl):
        v = vis[sl]
        gaps = []
        prev, start = v[0], 0
        for idx in range(1, len(v)):
            if v[idx] != prev:
                if not prev:
                    gaps.append(idx - start)
                prev, start = v[idx], idx
        if not prev:
            gaps.append(len(v) - start)
        return gaps

    alive_gaps = gap_hist(slice(11185, 18685))
    print("  all window gaps:", sorted(alive_gaps))
    for i, (a, b, k) in enumerate(segs):
        gaps = gap_hist(slice(a, b))
        print("  seg {:2d} {}: max_gap {:4d}  gaps>25: {:d}  gaps: {}".format(
            i, pm.LABELS[k], max(gaps) if gaps else 0, sum(1 for g in gaps if g > 25),
            sorted(gaps)))

    # rolling activity: 2 s mean of mean player speed
    act = (feats["mean_speed_0"] + feats["mean_speed_1"]) / 2.0
    act = np.where(np.isfinite(act), act, np.nan)
    w = 25
    c = np.concatenate(([0], np.nancumsum(np.where(np.isfinite(act), act, 0.0))))
    n = np.concatenate(([0], np.cumsum(np.isfinite(act))))
    idx = np.arange(len(act))
    low = np.maximum(0, idx - w)
    high = np.minimum(len(act), idx + w + 1)
    roll = (c[high] - c[low]) / np.maximum(1, n[high] - n[low])

    print("\nactivity around D boundaries:")
    for i, (a, b, k) in enumerate(segs):
        if k != 2:
            continue
        print("  D seg {:2d} [{:6d},{:6d}]: act p10/50/90 in seg {:.2f}/{:.2f}/{:.2f} | 5s before: {:.2f} | 5s after: {:.2f}".format(
            i, a, b, np.nanpercentile(roll[a:b], 10), np.nanmedian(roll[a:b]),
            np.nanpercentile(roll[a:b], 90),
            np.nanmean(roll[max(0, a - 125):a]), np.nanmean(roll[b:b + 125])))

    # rolling ball speed in 2 s window (median)
    bs = feats["ball_speed"]
    def roll_med(x, w):
        n = len(x)
        out = np.full(n, np.nan)
        for j in range(n):
            seg = x[max(0, j - w):min(n, j + w + 1)]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 3:
                out[j] = np.median(seg)
        return out

    rm = roll_med(bs, 25)
    print("\nrolling ball speed around D boundaries:")
    for i, (a, b, k) in enumerate(segs):
        if k != 2:
            continue
        print("  D seg {:2d}: in-seg p50 {:.2f} p90 {:.2f} | 2s before start {:.2f} | 2s after end {:.2f}".format(
            i, np.nanmedian(rm[a:b]), np.nanpercentile(rm[a:b], 90),
            np.nanmedian(rm[max(0, a - 50):a]), np.nanmedian(rm[b:b + 50])))


if __name__ == "__main__":
    main()
