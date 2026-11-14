"""Debug: what does dead_spans produce in the public window, and where are restarts?"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import half_video_blocks, require, source_frames  # noqa: E402
from work.model import (dead_spans, find_restart, load_tags, rolling,  # noqa: E402
                        interpolate_ball, KICK_RISE, KICK_WIN, ROLL_WIN, OUT_MARGIN,
                        ACT_LOW, BALL_SLOW, UNACC_DIST, D_SUSTAIN)
from work.load_atd import load_atd  # noqa: E402
from work.features import extract, team_vote_from_players  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402


def main():
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n = source_frames("tactical")
    ball, players = load_atd(raw_path, n)
    import xml.etree.ElementTree as ET
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team_of_slot = np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)
    feats = extract(ball, players, team_of_slot)
    tags = load_tags()

    vis = feats["ball_vis"] > 0.5
    bs_fast = rolling(np.where(vis, feats["ball_speed"], np.nan), KICK_WIN, "median", 3)
    bs_roll = rolling(np.where(vis, feats["ball_speed"], np.nan), ROLL_WIN, "median")
    act_roll = rolling((feats["mean_speed_0"] + feats["mean_speed_1"]) / 2.0, ROLL_WIN, "mean")

    # sanity: what does the state machine see around the kickoff?
    print("around kickoff 11185:")
    for f in range(11180, 11200):
        print("  f {:6d} vis {} bs_fast {:6.2f} bs_roll {:6.2f} act {:5.2f}".format(
            f, int(vis[f]), bs_fast[f], bs_roll[f], act_roll[f]))

    blocks = half_video_blocks()
    graded_ranges = [(a, b + 1) for _, a, b in blocks]
    dead = dead_spans(feats, tags, graded_ranges)
    sl = slice(11185, 18685)
    d = dead[sl]
    print("\ndead in window: {:.1f}%".format(100 * d.mean()))
    # print dead runs
    i = 0
    while i < len(d):
        if d[i]:
            j = i
            while j < len(d) and d[j]:
                j += 1
            print("  D run [{:6d},{:6d})".format(11185 + i, 11185 + j))
            i = j
        else:
            i += 1

    # GT segments for reference
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    runs = pm.valid_runs(gt)
    print("\nGT segments:")
    for a, b, k in pm.segments(gt, runs):
        print("  [{:6d},{:6d}) {}".format(a, b, pm.LABELS[k]))

    # where would find_restart fire if we scan each D run?
    print("\nrestart scan from D run starts:")
    f = 11185
    while f < 18685:
        if not dead[f]:
            f += 1
            continue
        b0 = f
        while f < 18685 and dead[f]:
            f += 1
        rst = find_restart(b0, f, vis, bs_fast, feats["ball_x"], feats["ball_y"], [t for t in tags if t["code"].upper() == "SET PIECES"])
        print("  D run [{:6d},{:6d}): restart at {}".format(b0, f, rst))


if __name__ == "__main__":
    main()
