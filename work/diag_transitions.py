"""Numeric diagnostics around each GT transition and D segment (understanding only)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from work.analyze import load_all  # noqa: E402
from work.features import nearest_team  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402


def smooth_code(codes, window):
    """-1/0/1 majority over centred window; -1 when no votes."""
    n = len(codes)
    out = np.full(n, -1, dtype=np.int8)
    idx = np.arange(n)
    for t in (0, 1):
        c = np.concatenate(([0], np.cumsum(codes == t)))
        low = np.maximum(0, idx - window)
        high = np.minimum(n, idx + window + 1)
        cnt = c[high] - c[low]
        if t == 0:
            c0 = cnt
        else:
            c1 = cnt
    ok = (c0 + c1) > 0
    out[ok] = (c1[ok] > c0[ok]).astype(np.int8)
    return out


def main():
    feats = load_all()
    nt = nearest_team(feats)
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    runs = pm.valid_runs(gt)
    trans = pm.transitions(gt, runs)
    # only transitions inside the public window
    trans = [t for t in trans if 11185 <= t[0] < 18685]

    print("GT transitions in window:")
    for f, frm, to in trans:
        print("  frame {:6d}  {} -> {}".format(f, pm.LABELS[frm], pm.LABELS[to]))

    print("\nsignal around each transition (frame offsets from GT transition):")
    bs = feats["ball_speed"]
    vis = feats["ball_vis"] > 0.5
    for f, frm, to in trans:
        a, b = f - 125, f + 125
        sl = slice(a, b)
        nt_s = nt[sl]
        # runs of nearest-team signal near the transition
        row = []
        for off in (-100, -50, -25, -10, 0, 10, 25, 50, 100):
            i = f + off
            row.append("{}:{}".format(off, nt[i]))
        bv = bs[sl].copy()
        bv[~vis[sl]] = np.nan
        q = [np.nanpercentile(bv, p) if np.isfinite(bv).any() else np.nan
             for p in (25, 50, 75)]
        print("  f={:6d} {}->{}  nt[off]: {}  ballSpeed p25/50/75: {:.1f}/{:.1f}/{:.1f}  vis {:.2f}".format(
            f, pm.LABELS[frm], pm.LABELS[to], " ".join(row),
            q[0], q[1], q[2], vis[sl].mean()))

    print("\nD segments: ball speed / activity / position profile")
    segs = pm.segments(gt, runs)
    for i, (a, b, k) in enumerate(segs):
        if k != 2:
            continue
        sl = slice(a, b)
        bv = bs[sl].copy()
        bv[~vis[sl]] = np.nan
        # rolling 2s median of ball speed
        def roll_med(x, w):
            out = np.full(len(x), np.nan)
            for j in range(w, len(x) - w):
                seg = x[j - w:j + w + 1]
                if np.isfinite(seg).sum() >= 5:
                    out[j] = np.nanmedian(seg)
            return out

        rm = roll_med(bv, 25)
        print("  D seg {:2d} f={:6d}-{:6d} len {:4d}".format(i, a, b, b - a))
        print("    ball vis {:5.2f}  |  rolling2s-ball-speed: min {:.2f} p50 {:.2f} max {:.2f}".format(
            vis[sl].mean(), np.nanmin(rm), np.nanmedian(rm), np.nanmax(rm)))
        for team in (0, 1):
            ms = feats["mean_speed_{}".format(team)][sl]
            print("    team {} mean player speed p10/50/90: {:.2f}/{:.2f}/{:.2f}".format(
                team, np.nanpercentile(ms, 10), np.nanmedian(ms), np.nanpercentile(ms, 90)))
        bx = feats["ball_x"][sl]
        by_ = feats["ball_y"][sl]
        near_edge = (bx < 3) | (bx > 102.3) | (by_ < 3) | (by_ > 65)
        print("    ball near edge (>3m inside) frac: {:.2f}  |  ball x p10/90: {:.1f}/{:.1f}  y p10/90: {:.1f}/{:.1f}".format(
            near_edge.mean(), np.nanpercentile(bx, 10), np.nanpercentile(bx, 90),
            np.nanpercentile(by_, 10), np.nanpercentile(by_, 90)))
        # what happens at boundaries: ball speed in the 2s before/after segment ends
        for side, (w0, w1) in (("start", (a - 50, a + 50)), ("end", (b - 50, b + 50))):
            w_sl = slice(*w0) if isinstance(w0, tuple) else None
            pass
        before = bs[a - 50:a]; after = bs[b:b + 50]
        bf = before[vis[a - 50:a]]
        af = after[vis[b:b + 50]]
        print("    2s before start: ball speed med {:.2f} vis {:.2f} | 2s after end: med {:.2f} vis {:.2f}".format(
            np.nanmedian(bf) if len(bf) else -1, vis[a - 50:a].mean(),
            np.nanmedian(af) if len(af) else -1, vis[b:b + 50].mean()))


if __name__ == "__main__":
    main()
