"""Fast-ish loader for the Metrica ATD txt into numpy arrays.

Format per line: "<frame>:<p0>;<p1>;...;<p45>:<ball>", each group "x,y,speed",
x/y normalised 0-1, speed m/s; a missing observation is "NaN,NaN,NaN".
frame counts from 1; video frame = frame_id - 1.
"""
import numpy as np


def parse_group(group):
    if group[0] == "N":
        return None
    vals = group.split(",")
    try:
        x = float(vals[0])
        y = float(vals[1])
        s = float(vals[2]) if vals[2] != "NaN" else np.nan
        return (x, y, s)
    except ValueError:
        return None


def load_atd(path, n_frames, n_slots=46, ball_only=False):
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")):
                parsed = parse_group(group)
                if parsed is not None:
                    players[frame, slot] = parsed
            parsed_ball = parse_group(parts[2])
            if parsed_ball is not None:
                ball[frame] = parsed_ball
    return ball, players
