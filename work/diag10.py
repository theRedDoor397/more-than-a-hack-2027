"""Compare vote variants per GT segment: weighted (sigma 2.5) vs k-nearest median."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import require, source_frames  # noqa: E402
from work.load_atd import load_atd  # noqa: E402
from work.features import team_vote_from_players  # noqa: E402
from work.model import interpolate_ball  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402


def k_nearest_vote(ball_int, players, team_of_slot, k=3, pitch=(105.3, 68.0)):
    """Per-frame: median team among the k nearest players to the ball.
    Returns logr-like: +-1 where decisive, 0 where split/unknown."""
    length, width = pitch
    n = len(ball_int)
    px = players[:, :, 0] * length
    py = players[:, :, 1] * width
    bx = ball_int[:, 0] * length
    by = ball_int[:, 1] * width
    vis = np.isfinite(bx)
    out = np.full(n, np.nan)
    for f in range(n):
        if not vis[f]:
            continue
        dx = px[f] - bx[f]
        dy = py[f] - by[f]
        d = np.hypot(dx, dy)
        order = np.argsort(d)
        teams = []
        for idx in order:
            if not np.isnan(d[idx]):
                teams.append(team_of_slot[idx])
                if len(teams) == k:
                    break
        if not teams:
            continue
        n_home = sum(1 for t in teams if t == 0)
        n_away = len(teams) - n_home
        # decisive when a clear majority
        if n_home == k:
            out[f] = 2.0
        elif n_away == k:
            out[f] = -2.0
        elif n_home > n_away:
            out[f] = 0.7
        elif n_away > n_home:
            out[f] = -0.7
        else:
            out[f] = 0.0
    return out


def main():
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    ball, players = load_atd(raw_path, source_frames("tactical"))
    import xml.etree.ElementTree as ET
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team_of_slot = np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)

    ball_int = interpolate_ball(ball)
    v0, v1, logr = team_vote_from_players(ball_int, players, team_of_slot, sigma=2.5)
    kn = k_nearest_vote(ball_int, players, team_of_slot, k=3)

    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    segs = pm.segments(gt, pm.valid_runs(gt))

    print("seg  label  len   weightedAgree  k3majorityAgree  k3signAgree")
    for i, (a, b, k) in enumerate(segs):
        if k >= 2:
            continue
        sl = slice(a, b)
        ok = np.isfinite(logr[sl])
        okk = np.isfinite(kn[sl])
        w_agree = np.mean((logr[sl][ok] > 0) == (k == 0)) if ok.any() else np.nan
        k_agree = np.mean((kn[sl][okk] > 0) == (k == 0)) if okk.any() else np.nan
        print("seg {:2d} {}  {:5d}   {:8.3f}       {:8.3f}        {:8.3f}".format(
            i, pm.LABELS[k], b - a, w_agree, k_agree, k_agree))


if __name__ == "__main__":
    main()
