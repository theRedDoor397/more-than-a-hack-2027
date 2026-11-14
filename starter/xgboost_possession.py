"""Compare XGBoost pseudo-label models without reading possession ground truth.

The teacher labels are generated independently for each tracking source by the published
nearest-player baseline. Broadcast records are additional training examples, not frame-aligned
features: the broadcast and tactical videos have different timelines.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter import baseline  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
FEATURE_COLUMNS = (
    "ball_visible", "ball_x", "ball_y", "ball_speed",
    "home_visible", "away_visible",
    "home_min_dist", "home_mean_dist", "home_median_dist", "home_std_dist",
    "away_min_dist", "away_mean_dist", "away_median_dist", "away_std_dist",
    "home_mean_speed", "away_mean_speed",
    "context_home_votes", "context_away_votes",
)
DEFAULT_WINDOWS = (25, 50, 125)
VALIDATION_BLOCK_FRAMES = 250


def team_meta(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams".format(meta_path))
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([0 if team_id == home_id else 1 for team_id in slots], dtype=np.int8)


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return (np.nan, np.nan, np.nan, np.nan)
    return (float(values.min()), float(values.mean()), float(np.median(values)),
            float(values.std()))


def build_feature_frame(raw_path, team_of_slot, pitch, n_frames):
    length, width = pitch
    rows = []
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue

            ball_fields = parts[2].split(",")
            if ball_fields[0].startswith("N"):
                ball_visible = 0
                ball_x = ball_y = ball_speed = np.nan
            else:
                ball_visible = 1
                ball_x = float(ball_fields[0])
                ball_y = float(ball_fields[1])
                ball_speed = float(ball_fields[2]) if ball_fields[2] != "NaN" else np.nan

            home_distances, away_distances = [], []
            home_speeds, away_speeds = [], []
            home_count = away_count = 0
            for slot, group in enumerate(parts[1].split(";")):
                fields = group.split(",")
                if fields[0].startswith("N"):
                    continue
                x, y = float(fields[0]), float(fields[1])
                speed = float(fields[2]) if fields[2] != "NaN" else np.nan
                if team_of_slot[slot] == 0:
                    distances, speeds = home_distances, home_speeds
                    home_count += 1
                else:
                    distances, speeds = away_distances, away_speeds
                    away_count += 1
                if ball_visible:
                    distances.append(float(np.hypot((x - ball_x) * length,
                                                    (y - ball_y) * width)))
                if not np.isnan(speed):
                    speeds.append(speed)

            home_stats = summarize(home_distances)
            away_stats = summarize(away_distances)
            rows.append({
                "frame": frame,
                "ball_visible": ball_visible,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_speed": ball_speed,
                "home_visible": home_count,
                "away_visible": away_count,
                "home_min_dist": home_stats[0],
                "home_mean_dist": home_stats[1],
                "home_median_dist": home_stats[2],
                "home_std_dist": home_stats[3],
                "away_min_dist": away_stats[0],
                "away_mean_dist": away_stats[1],
                "away_median_dist": away_stats[2],
                "away_std_dist": away_stats[3],
                "home_mean_speed": float(np.mean(home_speeds)) if home_speeds else np.nan,
                "away_mean_speed": float(np.mean(away_speeds)) if away_speeds else np.nan,
            })

    features = pd.DataFrame(rows)
    if len(features) != n_frames or not np.array_equal(
            features["frame"].to_numpy(), np.arange(n_frames)):
        raise ValueError("ATD must contain one ordered record for each of {} frames".format(
            n_frames))
    return features


def add_teacher_context(features, nearest_codes, window):
    home_cumulative = np.concatenate(([0], np.cumsum(nearest_codes == baseline.HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(nearest_codes == baseline.AWAY)))
    indexes = np.arange(len(nearest_codes))
    low = np.maximum(0, indexes - window)
    high = np.minimum(len(nearest_codes), indexes + window + 1)
    features["context_home_votes"] = home_cumulative[high] - home_cumulative[low]
    features["context_away_votes"] = away_cumulative[high] - away_cumulative[low]
    return features


def load_source(source, window):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, source=source)
    n_frames = source_frames(source, GAME)
    team_of_slot = team_meta(meta_path)
    features = build_feature_frame(raw_path, team_of_slot, GAMES[GAME]["pitch"], n_frames)
    nearest_codes = baseline.nearest_player_labels(
        raw_path, team_of_slot, n_frames, GAMES[GAME]["pitch"])
    features = add_teacher_context(features, nearest_codes, window)
    labels = baseline.smooth(nearest_codes, window)
    target = pd.Series(labels).map({"H": 0, "A": 1, "D": 2}).to_numpy(dtype=np.int8)
    return features, target


def make_model():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=2,
    )


def macro_f1(actual, predicted):
    scores = []
    for label in (0, 1, 2):
        true_positive = np.sum((actual == label) & (predicted == label))
        false_positive = np.sum((actual != label) & (predicted == label))
        false_negative = np.sum((actual == label) & (predicted != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))


def run_experiment(window, include_broadcast, tactical, broadcast, out_dir):
    datasets = [tactical] + ([broadcast] if include_broadcast else [])
    all_features = pd.concat([dataset[0] for dataset in datasets], ignore_index=True)
    all_targets = np.concatenate([dataset[1] for dataset in datasets])

    # Hold out recurring 10-second blocks from each timeline. This measures agreement with the
    # automatic teacher on unseen blocks, not accuracy against human ground truth.
    heldout = ((all_features["frame"].to_numpy() // VALIDATION_BLOCK_FRAMES) % 5) == 0
    if not np.any(heldout) or not np.any(~heldout):
        raise ValueError("Unable to form pseudo-label train/validation split")
    train_X = all_features.loc[~heldout, FEATURE_COLUMNS].fillna(-1)
    valid_X = all_features.loc[heldout, FEATURE_COLUMNS].fillna(-1)
    model = make_model()
    model.fit(train_X, all_targets[~heldout])
    valid_predictions = model.predict(valid_X)
    agreement = float(np.mean(valid_predictions == all_targets[heldout]))
    teacher_f1 = macro_f1(all_targets[heldout], valid_predictions)

    tactical_features = tactical[0].loc[:, FEATURE_COLUMNS].fillna(-1)
    tactical_predictions = model.predict(tactical_features)
    label_for = {0: "H", 1: "A", 2: "D"}
    labels = np.array([label_for[int(code)] for code in tactical_predictions], dtype="<U1")
    source_name = "tactical_broadcast" if include_broadcast else "tactical"
    out_path = os.path.join(out_dir, "xgboost_{}_w{}.csv".format(source_name, window))
    write_submission(out_path, labels)
    return out_path, agreement, teacher_f1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS),
                        help="fixed pseudo-label smoothing half-windows to compare")
    parser.add_argument("--out-dir", default=RESULTS_DIR,
                        help="where to write the tactical submission variants")
    args = parser.parse_args()
    if any(window < 0 for window in args.windows):
        parser.error("smoothing windows must be non-negative")

    out_dir = os.path.abspath(args.out_dir)
    tactical = {}
    broadcast = {}
    for window in sorted(set(args.windows)):
        tactical[window] = load_source("tactical", window)
        broadcast[window] = load_source("broadcast", window)

    print("Public/private possession ground truth is not read.")
    print("Held-out score is agreement with algorithm-generated pseudo-labels, not GT accuracy.")
    print("window  source                  teacher_agreement  teacher_macroF1  submission")
    for window in sorted(set(args.windows)):
        for include_broadcast in (False, True):
            path, agreement, teacher_f1 = run_experiment(
                window, include_broadcast, tactical[window], broadcast[window], out_dir)
            print("{:>6}  {:<23} {:>17.4f}  {:>15.4f}  {}".format(
                window, "tactical+broadcast" if include_broadcast else "tactical only",
                agreement, teacher_f1, os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
