"""Plot signals around the D segments and a few handovers (for understanding)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from work.analyze import load_all  # noqa: E402
from work.features import nearest_team  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT, exist_ok=True)


def gt_codes():
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    return pm.label_codes(pm.read_label_csv(gt_path))


def plot_window(feats, nt, gt, a, b, name):
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    t = np.arange(a, b) / 25.0
    sl = slice(a, b)

    ax = axes[0]
    bs = feats["ball_speed"][sl]
    vis = feats["ball_vis"][sl] > 0.5
    ax.plot(t, bs, lw=0.6, color="tab:blue", label="ball speed (m/s)")
    ax.scatter(t[vis], bs[vis], s=3, color="tab:blue", alpha=0.5)
    ax.set_ylabel("ball speed")
    ax.legend(loc="upper right")

    ax = axes[1]
    for team, color in ((0, "tab:red"), (1, "tab:orange")):
        ax.plot(t, feats["mean_speed_{}".format(team)][sl], lw=0.6, color=color,
                label="mean player speed team {}".format(team))
    ax.set_ylabel("player speed")
    ax.legend(loc="upper right")

    ax = axes[2]
    bx = feats["ball_x"][sl] / 105.3
    by_ = feats["ball_y"][sl] / 68.0
    ax.scatter(bx[vis], by_[vis], s=2, alpha=0.4, color="black")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("ball y (norm)")
    ax.set_title("ball position (norm); pitch boundary lines at 0 and 1")

    ax = axes[3]
    nt_s = nt[sl].astype(float)
    ax.scatter(t, nt_s, s=2, color="tab:purple", label="nearest team (0=H,1=A)")
    gt_s = gt[sl].astype(float)
    gt_s[gt_s == 3] = np.nan
    ax.plot(t, gt_s, lw=2, color="tab:green", label="GT label (H=0,A=1,D=2)")
    ax.set_ylabel("team / label")
    ax.set_xlabel("seconds from video start")
    ax.legend(loc="upper right")

    fig.suptitle(name)
    fig.tight_layout()
    path = os.path.join(OUT, name + ".png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print("wrote", path)


def main():
    feats = load_all()
    nt = nearest_team(feats)
    gt = gt_codes()

    windows = [
        ("D seg5 (5.5s)", 12665, 13100),
        ("D seg14 (20.4s)", 15750, 16500),
        ("D seg17 (18.3s)", 17250, 17950),
        ("D seg19 (14.5s)", 17900, 18500),
        ("handover seg9-13", 15300, 16000),
        ("handover seg0-2", 11180, 11650),
        ("kickoff", 11185, 11450),
        ("seg15 ball-missing alive", 16400, 17300),
    ]
    for name, a, b in windows:
        plot_window(feats, nt, gt, a, b, name.replace(" ", "_").replace("(", "").replace(")", ""))


if __name__ == "__main__":
    main()
