"""Debug: why does nearest-team say A on most of GT segment 3 (H, 11624-12665)?"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from work.analyze import load_all, CACHE  # noqa: E402
from work.features import nearest_team  # noqa: E402

feats = load_all()
nt = nearest_team(feats)
sl = slice(11624, 12665)
print("ball visible in seg3:", feats["ball_vis"][sl].mean())
nt_seg = nt[sl]
print("nearest-team counts: -1=%d  H=%d  A=%d" % (
    (nt_seg < 0).sum(), (nt_seg == 0).sum(), (nt_seg == 1).sum()))

d1h = feats["d1_0"][sl]
d1a = feats["d1_1"][sl]
print("median d1h=%.2f d1a=%.2f" % (np.nanmedian(d1h), np.nanmedian(d1a)))
both = np.isfinite(d1h) & np.isfinite(d1a)
print("frames with both teams visible: %d / %d" % (both.sum(), len(d1h)))
print("on those, A nearest: %.2f" % (d1a[both] < d1h[both]).mean())
print("d1h < 2m frac: %.2f   d1a < 2m frac: %.2f" % ((d1h[both] < 2).mean(), (d1a[both] < 2).mean()))
print("d1h < 5m frac: %.2f   d1a < 5m frac: %.2f" % ((d1h[both] < 5).mean(), (d1a[both] < 5).mean()))
# distribution of the min
for name, d in (("d1h", d1h), ("d1a", d1a)):
    q = [np.nanpercentile(d, p) for p in (10, 25, 50, 75, 90)]
    print(name, "percentiles 10/25/50/75/90:", " ".join("%.2f" % v for v in q))

# how many frames does each slot appear among the top-3 nearest to the ball?
from work.features import extract  # noqa: E402
# recompute per-slot distances
import sys as _s
_s.path.insert(0, os.path.join(ROOT))
