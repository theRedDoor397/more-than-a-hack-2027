"""Debug find_restart on the goal-kick span [18031, 18685)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import require, source_frames  # noqa: E402
from work.load_atd import load_atd  # noqa: E402
from work.features import extract  # noqa: E402
from work.model import (find_restart, load_tags, rolling,  # noqa: E402
                        KICK_WIN, KICK_RISE, KICK_TAG_MIN_SPEED, KICK_PRE_SLOW,
                        KICK_PRE_WIN, KICK_PRE_NEED, KICK_DISP, KICK_LOOK)


def main():
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    ball, players = load_atd(raw_path, source_frames("tactical"))
    import xml.etree.ElementTree as ET
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team_of_slot = np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)
    feats = extract(ball, players, team_of_slot)
    tags = load_tags()
    sp = [t for t in tags if t["code"].upper() == "SET PIECES"]

    vis = feats["ball_vis"] > 0.5
    bs_fast = rolling(np.where(vis, feats["ball_speed"], np.nan), KICK_WIN, "median", 3)
    bx, by = feats["ball_x"], feats["ball_y"]

    print("set piece tags covering 18000-18700:")
    for t in sp:
        if t["end"] > 18000 and t["start"] < 18700:
            print("  [{:6d},{:6d}) {:20s} side {}".format(t["start"], t["end"], t["type"], t["side"]))

    print("\nrestart scan result:", find_restart(18031, 18685, vis, bs_fast, bx, by, sp))

    print("\nper-second profile:")
    for s0 in range(18100, 18685, 50):
        sl = slice(s0, min(s0 + 50, 18685))
        v = vis[sl]
        if v.any():
            print("  {:6d}: vis {:2d}  bs_fast med {:5.2f} max {:5.2f}  ball ({:6.2f},{:6.2f})".format(
                s0, int(v.sum()), np.nanmedian(bs_fast[sl]), np.nanmax(bs_fast[sl]),
                bx[s0], by[s0]))
        else:
            print("  {:6d}: vis  0".format(s0))

    # simulate the event scan: print each candidate event and its confirmation
    print("\nevent scan:")
    prev_fast = False
    for t in range(18031, 18685):
        if not vis[t]:
            prev_fast = False
            continue
        fast = not np.isnan(bs_fast[t]) and bs_fast[t] > KICK_RISE
        if fast and not prev_fast:
            # confirmation details
            in_tag = any(tg["start"] <= t < tg["end"] for tg in sp)
            firm = not np.isnan(bs_fast[t]) and bs_fast[t] >= KICK_TAG_MIN_SPEED
            lo = max(0, t - KICK_PRE_WIN)
            rest = int(np.sum(vis[lo:t] & (bs_fast[lo:t] < KICK_PRE_SLOW))) + int(np.sum(~vis[lo:t]))
            x0, y0 = bx[t], by[t]
            best = 0.0
            for k in range(t + 1, min(18685, t + KICK_LOOK)):
                if vis[k]:
                    best = max(best, np.hypot(bx[k] - x0, by[k] - y0))
            print("  event f={:6d}  bs_fast={:5.2f}  in_tag={}  firm={}  rest={}/{}  disp={:.1f}".format(
                t, bs_fast[t], in_tag, firm, rest, KICK_PRE_WIN, best))
        prev_fast = fast


if __name__ == "__main__":
    main()
