"""A stronger possession heuristic for the Metrica ball-possession challenge.

This keeps the same nearest-player idea as the reference baseline, but handles short ball-occlusion
windows by carrying the previous possession through temporary gaps. That is a better fit for the
spellF1 metric, because a short missing-ball window inside an ongoing possession is much less costly
than splitting it into a dead-ball segment.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

HOME, AWAY, NOBODY = 0, 1, -1
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"


def team_of_each_track(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)


def nearest_player_labels(raw_path, team_of_track, n_frames, pitch):
    """Per-frame team of the player closest to the ball; NOBODY when the ball is missing."""
    length, width = pitch
    out = np.full(n_frames, NOBODY, dtype=np.int8)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            ball = parts[2]
            if ball[0] == MISSING:
                continue
            comma = ball.index(",")
            ball_x = float(ball[:comma]) * length
            ball_y = float(ball[comma + 1:ball.index(",", comma + 1)]) * width

            best, best_team = None, NOBODY
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                comma = group.index(",")
                dx = float(group[:comma]) * length - ball_x
                dy = float(group[comma + 1:group.index(",", comma + 1)]) * width - ball_y
                distance = dx * dx + dy * dy
                if best is None or distance < best:
                    best, best_team = distance, team_of_track[slot]

            out[int(parts[0]) - 1] = best_team
    return out


def majority_vote(codes, window):
    """Centred majority vote over nearby frames, ignoring frames with no player answer."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))

    home_cum = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cum = np.concatenate(([0], np.cumsum(codes == AWAY)))

    idx = np.arange(len(codes))
    low = np.maximum(0, idx - window)
    high = np.minimum(len(codes), idx + window + 1)
    home = home_cum[high] - home_cum[low]
    away = away_cum[high] - away_cum[low]

    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def carry_short_gaps(labels, max_gap):
    """If a team possession disappears for a short period, continue the previous team."""
    prev_label = "D"
    prev_index = -1
    out = labels.copy()
    for index, label in enumerate(out):
        if label == "D":
            if prev_label != "D" and index - prev_index <= max_gap:
                out[index] = prev_label
        else:
            prev_label = label
            prev_index = index
    return out


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))


def build_submission(out_path, window=75, gap=40):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    codes = nearest_player_labels(raw_path, team_of_each_track(meta_path), n_frames, GAMES[GAME]["pitch"])

    # Harder to split possessions if the ball is briefly occluded than to guess a dead ball during a
    # gap, so carry the last clear possession forward for a short time.
    raw_labels = np.full(n_frames, "D", dtype="<U1")
    prev_label = "D"
    prev_index = -1
    for index, code in enumerate(codes):
        if code == NOBODY:
            if prev_label != "D" and index - prev_index <= gap:
                raw_labels[index] = prev_label
            else:
                raw_labels[index] = "D"
        else:
            raw_labels[index] = "H" if code == HOME else "A"
            prev_label = raw_labels[index]
            prev_index = index

    # Majority vote suppresses flicker around handovers and player switches.
    vote_codes = np.full(n_frames, NOBODY, dtype=np.int8)
    for index, label in enumerate(raw_labels):
        if label == "H":
            vote_codes[index] = HOME
        elif label == "A":
            vote_codes[index] = AWAY

    labels = majority_vote(vote_codes, window)
    labels = carry_short_gaps(labels, gap)
    write_submission(out_path, labels)
    return labels


if __name__ == "__main__":
    path = os.path.join(RESULTS_DIR, "possession_stronger.csv")
    build_submission(path)
    print("wrote {}".format(os.path.relpath(path, ROOT)))
