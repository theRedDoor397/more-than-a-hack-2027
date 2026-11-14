"""Ball speed profile around the first-half kickoff (11185) and around D seg 14 end."""
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
    bs = feats["ball_speed"]
    bx = feats["ball_x"]
    by = feats["ball_y"]

    print("around 1st-half kickoff 11185:")
    for f in range(11175, 11200):
        print("  f {:6d}  vis {}  ball ({:6.2f},{:6.2f})  speed {:6.2f}".format(
            f, int(vis[f]), bx[f], by[f], bs[f]))

    print("\naround D seg14 end 16400:")
    for f in range(16375, 16415):
        print("  f {:6d}  vis {}  ball ({:6.2f},{:6.2f})  speed {:6.2f}".format(
            f, int(vis[f]), bx[f], by[f], bs[f]))

    print("\naround D seg17 start 17380:")
    for f in range(17365, 17395):
        print("  f {:6d}  vis {}  ball ({:6.2f},{:6.2f})  speed {:6.2f}".format(
            f, int(vis[f]), bx[f], by[f], bs[f]))


if __name__ == "__main__":
    main()
