"""Displacement + near-ball activity after the two untagged restart kicks (12980, 16400)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from work.analyze import load_all  # noqa: E402


def main():
    feats = load_all()
    vis = feats["ball_vis"] > 0.5
    bx = feats["ball_x"]
    by = feats["ball_y"]
    sp5h = feats["mean_speed_5_0"]
    sp5a = feats["mean_speed_5_1"]

    for name, t0 in (("free kick ~12980", 12980), ("throw-in ~16400", 16400),
                     ("kick-back pair ~16260 (seg14)", 16260),
                     ("corner ~17815", 17815), ("goal kick ~18355", 18355)):
        x0, y0 = bx[t0], by[t0]
        print("{}: ball at ({:.2f},{:.2f})".format(name, x0, y0))
        for dt in (25, 50, 75):
            f = t0 + dt
            d = np.hypot(bx[f] - x0, by[f] - y0) if vis[f] else float("nan")
            print("   +{:2d}f ({:.1f}s): ball ({:.2f},{:.2f}) vis {} disp {:.1f} m | act5 h/a {:.2f}/{:.2f}".format(
                dt, dt / 25, bx[f], by[f], int(vis[f]), d, sp5h[f], sp5a[f]))
        # max displacement over the next 75 frames and max near-ball activity
        dd = [np.hypot(bx[t0 + k] - x0, by[t0 + k] - y0) for k in range(1, 76) if vis[t0 + k]]
        a5 = [max(sp5h[t0 + k], sp5a[t0 + k]) for k in range(1, 76)]
        print("   max disp in 3 s: {:.1f} m | max act5 in 3 s: {:.2f}".format(
            max(dd) if dd else -1, max(a5) if a5 else -1))


if __name__ == "__main__":
    main()
