"""Exploration: per-GT-segment statistics of the tracking signals.

Purpose: understand what the corrected tracking means by "dead" and by "owning"
so the model can be designed from first principles. Aggregate diagnostics only;
no per-frame constants are derived from this.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import GAME, require, source_frames  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402
from work.load_atd import load_atd  # noqa: E402

PITCH = (105.3, 68.0)


def nearest_team_per_frame(ball, players, team_of_slot):
    """Team of the closest visible player to the ball; -1 when no answer."""
    n_frames = len(ball)
    out = np.full(n_frames, -1, dtype=np.int8)
    ball_vis = ~np.isnan(ball[:, 0])
    bx = ball[:, 0] * PITCH[0]
    by = ball[:, 1] * PITCH[1]
    for slot in range(players.shape[1]):
        x = players[:, slot, 0] * PITCH[0]
        y = players[:, slot, 1] * PITCH[1]
        vis = ball_vis & ~np.isnan(x)
        if not vis.any():
            continue
        d2 = np.full(n_frames, np.inf)
        d2[vis] = (x[vis] - bx[vis]) ** 2 + (y[vis] - by[vis]) ** 2
        better = d2 < np.full(n_frames, np.inf)
        out[better] = team_of_slot[slot]
        # track min distance to enable "better" comparison
        if not hasattr(nearest_team_per_frame, "mind2") or True:
            pass
    # slower but simple loop is fine for exploration
    return out


def main():
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    print("loading ATD ...")
    ball, players = load_atd(raw_path, n_frames)

    # team of each slot
    import xml.etree.ElementTree as ET
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team_of_slot = np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)
    print("teams:", [t.findtext("Name") for t in root.iter("Team")])
    print("ball visible:", np.sum(~np.isnan(ball[:, 0])), "/", n_frames)

    # GT segments in the public window
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    runs = pm.valid_runs(gt)
    segs = pm.segments(gt, runs)
    for i, (a, b, k) in enumerate(segs):
        print("GT seg {:2d}  frames {:6d}-{:6d}  len {:5d} ({:5.1f}s)  label {}".format(
            i, a, b, b - a, (b - a) / 25.0, pm.LABELS[k]))

    # per-segment stats
    print("\nseg  label  ballVis  meanBallSpeed  meanPlayerSpeed  nPlayers")
    for i, (a, b, k) in enumerate(segs):
        sl = slice(a, b)
        bv = np.mean(~np.isnan(ball[sl, 0]))
        bs = np.nanmean(ball[sl, 2])
        ps = np.nanmean(players[sl, :, 2])
        np_ = np.mean(np.isfinite(players[sl, :, 0]))
        print("{:3d}  {}     {:5.2f}   {:8.3f}      {:8.3f}       {:4.1f}".format(
            i, pm.LABELS[k], bv, bs, ps, np_))

    # smart tagging
    tag_path = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_smart_tagging.xml")
    events = None
    try:
        from kloppy import sportscode
        events = sportscode.load(tag_path).to_df()
    except Exception as e:
        print("kloppy failed:", e)
    if events is not None:
        events = events.copy()
        events["start_frame"] = (events["timestamp"].dt.total_seconds() * 25).round().astype(int)
        events["end_frame"] = (events["end_timestamp"].dt.total_seconds() * 25).round().astype(int)
        sp = events[events["code"].str.lower() == "set pieces"]
        print("\nset pieces (video frames):")
        for _, row in sp.iterrows():
            print("  {:5d}-{:5d}  {:16s}  {:22s}  {}".format(
                row["start_frame"], row["end_frame"], row["Type"], row["Team"], row["Half"]))


if __name__ == "__main__":
    main()
