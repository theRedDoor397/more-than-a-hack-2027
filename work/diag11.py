"""Feasibility: align broadcast ball trajectory to tactical via cross-correlation.

Tries the 8 dihedral transforms of the broadcast pitch coordinates and a coarse
offset grid; reports the best (transform, offset, agreement) for a test window.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import require, source_frames  # noqa: E402
from work.load_atd import load_atd  # noqa: E402


def transforms(x, y):
    """The 8 dihedral variants of pitch coordinates (normalised 0-1)."""
    out = []
    for rot in range(4):
        for flip in (False, True):
            a, b = x, y
            for _ in range(rot):
                a, b = b, 1.0 - a
            if flip:
                a = 1.0 - a
            out.append((a, b))
    return out


def main():
    # tactical ball (x, y) in metres
    t_ball, _ = load_atd(require(["atd_meta", "atd_raw"])[1],
                         source_frames("tactical"), 46, ball_only=True)
    b_ball, _ = load_atd(require(["atd_meta", "atd_raw"], source="broadcast")[1],
                         source_frames("broadcast"), 51, ball_only=True)
    tx = t_ball[:, 0] * 105.3
    ty = t_ball[:, 1] * 68.0
    tvis = np.isfinite(tx)

    # test window: the kickoff region in tactical frames [11185, 11500)
    w0, w1 = 11185, 11500
    tw = tvis[w0:w1]
    txw = tx[w0:w1]
    tyw = ty[w0:w1]

    best = []
    for ti, (bx, by) in enumerate(transforms(b_ball[:, 0] * 105.3, b_ball[:, 1] * 68.0)):
        bvis = np.isfinite(bx)
        for delta in range(-120000, 120001, 25):
            # tactical frame t corresponds to broadcast frame b = t + delta
            s0, s1 = w0 + delta, w1 + delta
            lo, hi = max(0, s0), min(len(bx), s1)
            if hi - lo < 100:
                continue
            tlo, thi = max(w0, lo - delta), min(w1, hi - delta)
            lo, hi = tlo + delta, thi + delta
            m = bvis[lo:hi] & tvis[tlo:thi]
            if m.sum() < 50:
                continue
            d = np.hypot(bx[lo:hi][m] - tx[tlo:thi][m],
                         by[lo:hi][m] - ty[tlo:thi][m])
            score = float(np.mean(d))
            best.append((score, ti, delta, int(m.sum())))
    best.sort()
    print("best 10 (mean |ball diff| metres, transform, delta, n):")
    for score, ti, delta, n in best[:10]:
        print("  {:.3f}  transform {:d}  delta {:>8}  n {:4d}".format(score, ti, delta, n))


if __name__ == "__main__":
    main()
