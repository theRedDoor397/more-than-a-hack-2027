"""Refine broadcast->tactical offset and check for drift and frame-rate mismatch."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import require, source_frames  # noqa: E402
from work.load_atd import load_atd  # noqa: E402

L, W = 105.3, 68.0


def load():
    t_ball, _ = load_atd(require(["atd_meta", "atd_raw"])[1],
                         source_frames("tactical"), 46)
    b_ball, _ = load_atd(require(["atd_meta", "atd_raw"], source="broadcast")[1],
                         source_frames("broadcast"), 51)
    tx = t_ball[:, 0] * L
    ty = t_ball[:, 1] * W
    bx = b_ball[:, 0] * L
    by = b_ball[:, 1] * W
    return (tx, ty, np.isfinite(tx)), (bx, by, np.isfinite(bx))


def score_at(t, b, delta, w0, w1):
    tx, ty, tvis = t
    bx, by, bvis = b
    s0, s1 = w0 + delta, w1 + delta
    lo, hi = max(0, s0), min(len(bx), s1)
    tlo, thi = max(w0, lo - delta), min(w1, hi - delta)
    lo, hi = tlo + delta, thi + delta
    m = bvis[lo:hi] & tvis[tlo:thi]
    if m.sum() < 30:
        return None, 0
    d = np.hypot(bx[lo:hi][m] - tx[tlo:thi][m], by[lo:hi][m] - ty[tlo:thi][m])
    return float(np.median(d)), int(m.sum())


def main():
    t, b = load()
    # refine around -4000 for the kickoff window
    print("refine around -4000 (window 11185-11500):")
    for delta in range(-4150, -3850, 10):
        s, n = score_at(t, b, delta, 11185, 11500)
        print("  delta {:>7}: median {:.2f} (n={})".format(delta, s if s else -1, n))

    # check other windows with the same offset
    print("\nother windows at delta -4000 (and -4020):")
    for w0 in (12000, 14000, 16000, 17900):
        s1 = w0 + 600
        for delta in (-4000, -4020):
            s, n = score_at(t, b, delta, w0, s1)
            print("  win {} delta {:>7}: median {:.2f} (n={})".format(w0, delta, s if s else -1, n))

    # is there drift? search a wider grid per window
    print("\nper-window best offset (grid 100):")
    for w0 in (12000, 14000, 16000, 17900, 25000, 35000, 50000, 70000):
        s1 = w0 + 400
        best = None
        for delta in range(-4200, -3700, 100):
            s, n = score_at(t, b, delta, w0, s1)
            if s is not None and (best is None or s < best[0]):
                best = (s, delta, n)
        print("  win {}: {}".format(w0, best))


if __name__ == "__main__":
    main()
