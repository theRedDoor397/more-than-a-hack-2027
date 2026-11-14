"""Compare model segments vs GT segments in the public window."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import half_video_blocks, GAME  # noqa: E402
from work.model import build_submission, load_tags, dead_spans  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402


def segs_from(labels, a, b):
    out = []
    f = a
    while f < b:
        start = f
        lab = labels[f]
        while f < b and labels[f] == lab:
            f += 1
        out.append((start, f, lab))
    return out


def main():
    labels, dead, logr = build_submission(
        os.path.join(ROOT, "evaluation", "results", "model_v1.csv"),
        cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))

    a, b = 11185, 18685
    gt_segs = [s for s in pm.segments(gt, pm.valid_runs(gt)) if s[0] < b and s[1] > a]
    my_segs = segs_from(labels, a, b)

    print("GT segments:")
    for s, e, k in gt_segs:
        print("  [{:6d},{:6d}) {:5d}f {}".format(s, e, e - s, pm.LABELS[k]))
    print("model segments:")
    for s, e, k in my_segs:
        print("  [{:6d},{:6d}) {:5d}f {}".format(s, e, e - s, k))

    # where does the model differ? mark error types per frame
    err = np.zeros(b - a, dtype=np.int8)
    for s, e, k in gt_segs:
        for f in range(max(a, s), min(b, e)):
            gl = gt[f]
            ml = labels[f]
            if gl != ml:
                err[f - a] = 1
    print("\nerror rate {:.3f}".format(err.mean()))

    # count team flips inside GT alive spans (fragmentation source)
    n_flips_inside = 0
    for s, e, k in gt_segs:
        if k == 2:
            continue
        prev = None
        for f in range(s, e):
            ml = labels[f]
            if ml in ("H", "A") and ml != prev and prev is not None:
                n_flips_inside += 1
            prev = ml
    print("team flips inside GT alive spans: {}".format(n_flips_inside))


if __name__ == "__main__":
    main()
