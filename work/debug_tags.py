"""All smart-tagging instances overlapping the public window, and team agreement."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from work.model import load_tags  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402


def main():
    tags = load_tags()
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))

    a, b = 11185, 18685
    print("tags overlapping the window:")
    for t in tags:
        if t["end"] > a and t["start"] < b:
            s, e = max(a, t["start"]), min(b, t["end"])
            # GT majority label inside the overlap
            lab = pm.LABELS[np.bincount(gt[s:e], minlength=4)[:3].argmax()]
            side = "H" if t["side"] == 0 else ("A" if t["side"] == 1 else "-")
            agree = ""
            if t["side"] >= 0 and lab in ("H", "A"):
                agree = "OK" if side == lab else "DISAGREE"
            print("  [{:6d},{:6d}) {:32s} {:12s} side {}  GT-majority {} {}".format(
                s, e, t["code"], t["type"], side, lab, agree))


if __name__ == "__main__":
    main()
