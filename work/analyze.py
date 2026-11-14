"""Compare tracking features against the public GT structure (understanding only).

Prints per-segment feature summaries and distributions to see which signals
discriminate D from alive, and H from A.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import GAME, require, source_frames  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402
from work.load_atd import load_atd  # noqa: E402
from work.features import extract, nearest_team  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def load_all():
    feat_path = os.path.join(CACHE, "tactical_feats.npz")
    if os.path.isfile(feat_path):
        data = np.load(feat_path, allow_pickle=False)
        return {k: data[k] for k in data.files}
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    print("loading ATD ...")
    ball, players = load_atd(raw_path, n_frames)
    import xml.etree.ElementTree as ET
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team_of_slot = np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)
    feats = extract(ball, players, team_of_slot)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(feat_path, **feats)
    return feats


def main():
    feats = load_all()
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    runs = pm.valid_runs(gt)
    segs = pm.segments(gt, runs)

    nt = nearest_team(feats)

    print("seg  label  len_s  ballVis  bs_med  bs_p90  ps_med  d1h_med  d1a_med  ntAgr(ball)  ntAgr(all)")
    for i, (a, b, k) in enumerate(segs):
        sl = slice(a, b)
        bv = np.mean(feats["ball_vis"][sl])
        bs = feats["ball_speed"][sl]
        d1h = feats["d1_0"][sl]
        d1a = feats["d1_1"][sl]
        ps = np.nanmean([feats["mean_speed_0"][sl], feats["mean_speed_1"][sl]])
        with np.errstate(all="ignore"):
            bs_med = np.nanmedian(bs)
            bs_p90 = np.nanpercentile(bs, 90)
            d1h_med = np.nanmedian(d1h)
            d1a_med = np.nanmedian(d1a)
        nt_ok = nt[sl] >= 0
        agree = np.mean(nt[sl][nt_ok] == k) if nt_ok.any() and k < 2 else np.nan
        agree_all = np.mean((nt[sl] == k) | (nt[sl] < 0)) if k < 2 else np.nan
        print("{:3d}  {}     {:5.1f}   {:6.2f}  {:6.2f}  {:6.2f}  {:5.2f}  {:6.2f}   {:6.2f}    {:7.3f}      {:7.3f}".format(
            i, pm.LABELS[k], (b - a) / 25.0, bv, bs_med, bs_p90, ps, d1h_med, d1a_med,
            agree, agree_all))

    # distribution comparison D vs alive on key features
    print("\nfeature               D-median  alive-median")
    for key in ("ball_speed", "mean_speed_0", "mean_speed_1", "max_speed_0", "max_speed_1",
                "d1_0", "d1_1", "n5_0", "n5_1"):
        vals = feats[key]
        d_vals = np.concatenate([vals[a:b] for a, b, k in segs if k == 2])
        al_vals = np.concatenate([vals[a:b] for a, b, k in segs if k != 2])
        with np.errstate(all="ignore"):
            print("{:20s}  {:8.2f}  {:8.2f}".format(
                key, np.nanmedian(d_vals), np.nanmedian(al_vals)))

    # run-length structure of the nearest-team signal in the window
    print("\nnearest-team run lengths (frames) in window:")
    nt_win = nt[11185:18685]
    runs_nt = []
    prev, start = nt_win[0], 0
    for idx in range(1, len(nt_win)):
        if nt_win[idx] != prev:
            runs_nt.append((prev, idx - start))
            prev, start = nt_win[idx], idx
    runs_nt.append((prev, len(nt_win) - start))
    from collections import Counter
    hist = Counter()
    for code, length in runs_nt:
        hist[(code, min(length, 60))] += 1
    total_runs = len(runs_nt)
    print("total runs:", total_runs)
    for (code, length), count in sorted(hist.items(), key=lambda x: (x[0][0], x[0][1])):
        print("  team {}  len {:>3}  n={}".format(code, length, count))


if __name__ == "__main__":
    main()
