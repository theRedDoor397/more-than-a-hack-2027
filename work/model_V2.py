"""Possession model: dead-ball state machine + team-vote decoding.

Two coupled state machines over the graded frames:
  * alive/dead -- dead entered on ball-out, whistle signatures (players stopped,
    ball not being played), long tracker gaps; exited on a confirmed restart kick.
  * team (H/A) -- hysteresis decode of a distance-weighted player vote around the
    ball, inside alive spans, with a kick-team prior at tagged restarts.

All parameters are chosen from domain reasoning (speeds, distances, durations in
physical units), not fitted to any ground truth. Produces a full 199,319-row
submission CSV.
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from work.load_atd import load_atd  # noqa: E402
from work.features import extract, team_vote_from_players  # noqa: E402

PITCH = GAMES[GAME]["pitch"]
LENGTH, WIDTH = PITCH
FPS = 25

# --- parameters (principled values) ------------------------------------------
VOTE_SIGMA = 2.5       # m: a controlling player is ~1-2 m from the ball
VOTE_SMOOTH = 10       # frames (0.4 s) centred smoothing of the vote log-ratio
FLIP_LOGIT = 2.5       # |vote log-ratio| (~12:1) to change team: decisive evidence
FLIP_HOLD = 50         # frames (2 s): the new team's evidence must persist after
FLIP_HOLD_FRAC = 0.7   #   the crossing for the change of possession to commit
FLIP_BALL_SPEED = 7.0  # m/s: no controlled change while the ball is in flight
RESTART_SET = 25       # frames (1 s): window after a restart to read the kicker
RESTART_SET_DIST = 2.0  # m: the ball must be at a player's feet for the kicker read
BALL_SLOW = 1.5        # m/s rolling median: a ball this slow is not being played
ACT_LOW = 1.5          # m/s rolling mean player speed: below = nobody is playing
ACT_ACTIVE = 2.5       # m/s: above = players clearly playing (invisible-ball escape)
ACT_ACTIVE_SUSTAIN = 100  # frames (4 s) of activity to escape D with the ball lost
UNACC_DIST = 6.0       # m: ball farther than this from every player = loose
D_SUSTAIN = 33         # frames (1.3 s) of dead evidence to enter D
OUT_MARGIN = 1.0       # m beyond the pitch lines = out of play
OUT_SUSTAIN = 5        # frames to confirm out (tracker noise near lines)
KICK_RISE = 3.5        # m/s: a kicked ball's short-window speed exceeds this
KICK_WIN = 9           # frames short rolling window for kick events
KICK_TAG_MIN_SPEED = 5.0  # m/s: a tag-confirmed restart must be a firm kick
KICK_PRE_WIN = 25      # frames: length of the preceding-at-rest check
KICK_PRE_MED = 2.0     # m/s: median of the preceding second must be below this
                       #   (a restart kick is struck from a ball genuinely at rest)
KICK_DISP = 15.0       # m net displacement within 3 s = play has restarted
KICK_DISP_OPEN = 20.0  # m: open-field events need more travel...
KICK_END_FAST = 2.5    # m/s: ...and the ball must still be moving at the window
                       # end (a kick-back is rolled to a standstill)
KICK_LINE = 3.0        # m from a boundary line = a throw-in restart position
KICK_PITCH_MARGIN = 2.0  # m: a tag-confirmed restart may be taken this far past
                       #   the lines (corners at the flag, tracker overshoot)
KICK_BEYOND_MIN_SPEED = 8.0  # m/s: a tag-confirmed kick from beyond the line
                       #   must be this firm (a corner, not a keeper's jog-back)
KICK_LOOK = 75         # frames (3 s) look-ahead for the displacement check
GAP_INTERP = 50        # frames: interpolate ball position over gaps up to this
TAG_PRIOR = 50         # frames (2 s) of kick-team bias after a tagged restart
TAG_PRIOR_STRENGTH = 1.2  # logit bias toward the tagged kicking team
ROLL_WIN = 51          # frames (~2 s) rolling window for speed/activity

HOME, AWAY = 0, 1


# --- data loading ------------------------------------------------------------

def load_team_slots(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    names = [t.findtext("Name") for t in root.iter("Team")]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team_of_slot = np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)
    return team_of_slot, names


def _norm_name(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def load_tags(team_names):
    """Smart tagging instances: {code, start, end, side, type} sorted by start.

    Tag team names are mapped to HOME/AWAY via the ATD metadata team order
    (team_names[0] = HOME), not a hard-coded table, so the tag side always
    agrees with the vote sign. The tagger's spelling can differ from the
    metadata ("Atletic Club" vs "Athletic Club"), so match fuzzily and report
    anything left unmatched.
    """
    import difflib
    keys = [_norm_name(n) for n in team_names]
    cache = {}

    def side_of(name):
        if name is None:
            return -1
        if name not in cache:
            hit = difflib.get_close_matches(_norm_name(name), keys, n=1, cutoff=0.6)
            cache[name] = keys.index(hit[0]) if hit else -1
            if not hit:
                print("WARNING: tag team {!r} matches no metadata team {}".format(
                    name, team_names))
        return cache[name]

    events = []
    path = require(["smart_tagging"], GAME)[0]
    for instance in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text")
                for lab in instance.findall("label")}
        events.append({
            "code": (instance.findtext("code") or "").strip(),
            "start": int(round(float(instance.findtext("start")) * FPS)),
            "end": int(round(float(instance.findtext("end")) * FPS)),
            "side": side_of(tags.get("Team")),
            "type": (tags.get("Type") or "").strip(),
        })
    events.sort(key=lambda e: e["start"])
    print("tag teams: {}".format({k: v for k, v in cache.items()}))
    return events


# --- signals -----------------------------------------------------------------

def rolling(signal, win, how="median", min_periods=5):
    s = pd.Series(signal)
    roll = s.rolling(win, center=True, min_periods=min_periods)
    result = roll.median() if how == "median" else roll.mean()
    return np.array(result.to_numpy(), copy=True)


def interpolate_ball(ball, max_gap=GAP_INTERP):
    """Linearly interpolate ball x/y across invisible gaps <= max_gap frames."""
    x = ball[:, 0].copy()
    y = ball[:, 1].copy()
    vis = np.isfinite(x)
    n = len(x)
    i = 0
    while i < n:
        if not vis[i]:
            j = i
            while j < n and not vis[j]:
                j += 1
            gap = j - i
            if 0 < gap <= max_gap and i > 0 and j < n:
                x[i:j] = np.linspace(x[i - 1], x[j], gap + 2)[1:-1]
                y[i:j] = np.linspace(y[i - 1], y[j], gap + 2)[1:-1]
            i = j
        else:
            i += 1
    out = ball.copy()
    out[:, 0] = x
    out[:, 1] = y
    return out


# --- dead-ball state machine -------------------------------------------------

def dead_spans(feats, tags, graded_ranges):
    """Run the alive/dead state machine; returns boolean dead[] per frame.

    Dead evidence per frame (rolling signals, ~2 s windows):
      ball_quiet  = ball invisible OR ball speed < BALL_SLOW
      unacc       = ball visible but > UNACC_DIST m from every player
      out_pitch   = ball beyond the pitch lines by OUT_MARGIN
      act_quiet   = mean player speed < ACT_LOW
    Alive -> dead: out_pitch for OUT_SUSTAIN frames, or (ball_quiet|unacc) &
    act_quiet sustained D_SUSTAIN frames.
    Dead -> alive: a confirmed restart kick (see _confirm_kick), or an
    activity escape when the tracker has lost the ball but the players are
    clearly playing.
    Between a GOALS tag and the following kick-off tag the ball is dead; before
    the kick-off tag interval of a half it is dead (teams walking out).
    """
    n = len(feats["ball_vis"])
    vis = feats["ball_vis"] > 0.5
    bx = feats["ball_x"]
    by = feats["ball_y"]
    d1 = np.minimum(feats["d1_0"], feats["d1_1"])

    bs_roll = rolling(np.where(vis, feats["ball_speed"], np.nan), ROLL_WIN, "median")
    act_roll = rolling((feats["mean_speed_0"] + feats["mean_speed_1"]) / 2.0,
                       ROLL_WIN, "mean")
    bs_fast = rolling(np.where(vis, feats["ball_speed"], np.nan), KICK_WIN, "median",
                      min_periods=3)

    out_pitch = (bx < -OUT_MARGIN) | (bx > LENGTH + OUT_MARGIN) | \
                (by < -OUT_MARGIN) | (by > WIDTH + OUT_MARGIN)

    # forced-dead windows: goal celebrations (GOALS tag -> next kick-off tag).
    # A GOALS tag is a wide window around the goal, so it can END after the
    # kick-off tag that follows it has already STARTED: pair on the goal's
    # start, not its end.
    goals = [t for t in tags if t["code"].upper() == "GOALS"]
    kickoff_tags = [t for t in tags if t["code"].upper() == "SET PIECES"
                    and t["type"].lower().startswith("kick off")]
    forced = []
    for g in goals:
        after = [k for k in kickoff_tags if k["start"] >= g["start"]]
        forced.append((g["start"], (after[0]["start"] if after else g["end"] + 45 * FPS)))
    forced.sort()

    set_piece_tags = [t for t in tags if t["code"].upper() == "SET PIECES"]

    dead = np.zeros(n, dtype=bool)
    # iterate the graded blocks; treat everything else as dead (label X anyway)
    for a, b in graded_ranges:
        # forced dead until the half-opening kick-off tag of this block
        block_kickoff = next((t for t in kickoff_tags
                              if t["type"].lower() == "kick off (start)"
                              and a <= t["start"] < b), None)

        alive = False  # each half starts dead, before the kick-off
        sustain = 0
        out_run = 0
        # find_restart(f, lim) == None implies None for every later f' < lim
        # (the candidate kicks from f' are a subset of those from f), so skip
        # the rescan until the search limit changes. Without this a long dead
        # stretch rescans to its end from every frame (quadratic).
        no_restart_until = a

        f = a
        while f < b:
            pre_kickoff = block_kickoff is not None and f < block_kickoff["start"]
            in_forced = any(s <= f < e for s, e in forced)
            # a restart must not be found inside a later forced-dead window
            # (goal celebration): bound the search at the next window's start
            limit = min([b] + [s for s, _ in forced if s > f])

            if alive:
                # ---- alive -> dead
                if out_pitch[f] and vis[f]:
                    out_run += 1
                else:
                    out_run = 0

                bs_roll_f = bs_roll[f]
                ball_quiet = (not vis[f]) or (bs_roll_f < BALL_SLOW
                                              if not np.isnan(bs_roll_f) else True)
                unacc = vis[f] and d1[f] > UNACC_DIST
                act_quiet = (not np.isnan(act_roll[f])) and act_roll[f] < ACT_LOW
                dead_ev = (ball_quiet or unacc) and act_quiet

                if out_run >= OUT_SUSTAIN:
                    alive = False
                    sustain = 0
                elif dead_ev:
                    sustain += 1
                    if sustain >= D_SUSTAIN:
                        alive = False
                        sustain = 0
                else:
                    sustain = 0
                dead[f] = not alive
                f += 1
            else:
                # ---- dead -> alive: a confirmed restart kick, or an activity
                # escape (the ball lost by the tracker while play clearly goes
                # on: players running for several seconds)
                restart_at = None
                if not pre_kickoff and not in_forced:
                    if f >= no_restart_until:
                        restart_at = find_restart(f, limit, vis, bs_fast, bs_roll,
                                                  bx, by, set_piece_tags)
                        if restart_at is None:
                            no_restart_until = limit
                    if restart_at is None and not vis[f]:
                        # ball invisible: look ahead for a sustained active stretch
                        look = 0
                        k = f
                        while k < limit and not vis[k]:
                            if not np.isnan(act_roll[k]) and act_roll[k] > ACT_ACTIVE:
                                look += 1
                            else:
                                look = 0
                            if look >= ACT_ACTIVE_SUSTAIN:
                                restart_at = k - look + 1
                                break
                            k += 1
                if restart_at is None:
                    dead[f] = True
                    f += 1
                else:
                    dead[f:restart_at] = True
                    alive = True
                    sustain = out_run = 0
                    f = restart_at
    return dead


def find_restart(f, b, vis, bs_fast, bs_roll, bx, by, set_piece_tags):
    """First confirmed restart kick at or after frame f (inside [f, b)), else None.

    A kick event: ball speed (short rolling median) rising from slow/invisible to
    > KICK_RISE. The scan starts in the state the ball is in at f, so a ball
    already rolling when the dead stretch began never counts as a kick.
    Confirmed when the ball moves >= KICK_DISP metres (net) within KICK_LOOK
    frames (play has restarted), or it is a firm kick from rest inside a
    set-piece tag.
    """
    if vis[f]:
        prev_fast = not np.isnan(bs_fast[f]) and bs_fast[f] > KICK_RISE
    else:
        prev_fast = False
    t = f
    while t < b:
        if not vis[t]:
            prev_fast = False
            t += 1
            continue
        fast = not np.isnan(bs_fast[t]) and bs_fast[t] > KICK_RISE
        if fast and not prev_fast:
            if _confirm_kick(t, b, vis, bs_fast, bs_roll, bx, by, set_piece_tags):
                return t
        prev_fast = fast
        t += 1
    return None


def _confirm_kick(t, b, vis, bs_fast, bs_roll, bx, by, set_piece_tags):
    """A kick at t is a restart iff the ball travels far enough within KICK_LOOK
    frames and play visibly continues, or it is a firm kick from a ball at rest
    inside a set-piece tag.

    Throw-in restarts happen at the touchline, so events within KICK_LINE of a
    boundary need only KICK_DISP m of travel. Open-field events need KICK_DISP_OPEN
    m AND the ball still moving at the end of the window: a stoppage kick-back is
    rolled to a standstill, a restart is followed by play.
    """
    x0, y0 = bx[t], by[t]
    best = 0.0
    end = min(b, t + KICK_LOOK)
    for k in range(t + 1, end):
        if vis[k]:
            d = np.hypot(bx[k] - x0, by[k] - y0)
            if d > best:
                best = d
    near_line = (x0 < KICK_LINE) | (x0 > LENGTH - KICK_LINE) | \
                (y0 < KICK_LINE) | (y0 > WIDTH - KICK_LINE)
    if near_line:
        if best >= KICK_DISP:
            return True
    else:
        end_fast = not np.isnan(bs_roll[end - 1]) and bs_roll[end - 1] > KICK_END_FAST
        if best >= KICK_DISP_OPEN and end_fast:
            return True
    firm_peak = bs_fast[t]
    for k in range(t + 1, min(b, t + 10)):
        if not np.isnan(bs_fast[k]) and bs_fast[k] > firm_peak:
            firm_peak = bs_fast[k]
    # the ball must genuinely have been at rest before the kick: median of the
    # preceding second's speeds (invisible counts as at rest)
    lo = max(0, t - KICK_PRE_WIN)
    prev_speeds = np.where(vis[lo:t], bs_fast[lo:t], 0.0)
    at_rest = bool(np.nanmedian(prev_speeds) < KICK_PRE_MED)
    # a restart kick is struck from (roughly) the field of play; corners sit at
    # the flag and the tracker overshoots the lines, so allow a small margin,
    # but a kick struck from beyond the line must be firm (a corner, not a
    # keeper jogging the ball back into play)
    beyond = (x0 < -OUT_MARGIN) or (x0 > LENGTH + OUT_MARGIN) or \
        (y0 < -OUT_MARGIN) or (y0 > WIDTH + OUT_MARGIN)
    on_pitch = not beyond or ((x0 >= -KICK_PITCH_MARGIN)
                              and (x0 <= LENGTH + KICK_PITCH_MARGIN)
                              and (y0 >= -KICK_PITCH_MARGIN)
                              and (y0 <= WIDTH + KICK_PITCH_MARGIN))
    min_firm = KICK_TAG_MIN_SPEED if not beyond else KICK_BEYOND_MIN_SPEED
    firm = not np.isnan(firm_peak) and firm_peak >= min_firm
    if firm and at_rest and on_pitch:
        for tag in set_piece_tags:
            if tag["start"] <= t < tag["end"]:
                return True
    # position-anchored restarts (kick-offs, corners, goal kicks) are labelled
    # reliably by the tagger; the kick itself is often invisible to the tracker
    # (ball lost at the flag / in flight), so any struck ball inside such a tag
    # after a rest confirms the restart. Free kicks and throw-ins get no such
    # leniency: their tags are wide and their setups include kick-backs.
    if at_rest:
        for tag in set_piece_tags:
            if tag["start"] <= t < tag["end"]:
                kind = tag["type"].lower()
                if kind.startswith("kick off") or kind.startswith("corner") \
                        or kind.startswith("goal kick"):
                    return True
    return False


# --- broadcast second opinion --------------------------------------------------

# The broadcast ATD runs on its own timeline; measured by aligning the two ball
# trajectories (mean |error| ~1.4 m at this offset, stable across the match, no
# drift detected). Broadcast pitch coordinates share the tactical frame (the
# identity transform aligned best of all 8 dihedrals).
BCAST_OFFSET = 4010      # tactical frame t = broadcast frame b + this
BCAST_KEEP_DIST = 6.0    # m: aligned ball positions must agree this well


def aligned_broadcast_vote(tball, cache_dir=None):
    """The broadcast vote re-indexed onto tactical frames (NaN where none).

    The two tracking pipelines are independent (different cameras, different
    detections), so their votes are two observations of the same event.
    Broadcast frames that do not align to the tactical ball (replays, cuts,
    tracker loss) contribute nothing.
    """
    n = len(tball)

    bvote_path = os.path.join(cache_dir, "bvote_{}_s{}_o{}_k{}.npz".format(
        GAME, VOTE_SIGMA, BCAST_OFFSET, BCAST_KEEP_DIST)) if cache_dir else None
    if bvote_path and os.path.isfile(bvote_path):
        data = np.load(bvote_path, allow_pickle=False)
        return data["bvote"].copy()
    b_ball, b_players = load_atd(
        require(["atd_meta", "atd_raw"], GAME, source="broadcast")[1],
        source_frames("broadcast", GAME), 51)
    b_team = load_team_slots(
        require(["atd_meta", "atd_raw"], GAME, source="broadcast")[0])[0]
    b_int = interpolate_ball(b_ball)
    _, _, b_logr = team_vote_from_players(b_int, b_players, b_team,
                                          sigma=VOTE_SIGMA)
    btx = b_ball[:, 0] * LENGTH
    bty = b_ball[:, 1] * WIDTH
    ttx = tball[:, 0] * LENGTH
    tty = tball[:, 1] * WIDTH
    tvis = np.isfinite(ttx)
    bvote = np.full(n, np.nan)
    for b in range(len(b_logr)):
        t = b + BCAST_OFFSET
        if not 0 <= t < n or not (np.isfinite(b_logr[b]) and tvis[t]):
            continue
        if np.hypot(btx[b] - ttx[t], bty[b] - tty[t]) < BCAST_KEEP_DIST:
            bvote[t] = b_logr[b]
    if bvote_path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(bvote_path, bvote=bvote)
    return bvote


# --- team decoding -----------------------------------------------------------

def decode_teams(logr_smoothed, dead, graded_ranges, bs_roll=None, d1=None,
                 restart_edges=None):
    """Sustained-evidence decode of the vote inside alive spans, per half.

    A change of possession commits only when the challenger's vote crosses
    +-FLIP_LOGIT while the ball is not in flight (rolling speed below
    FLIP_BALL_SPEED), and persists on the new side for the next FLIP_HOLD frames
    (>= FLIP_HOLD_FRAC of them). The flip is placed at the crossing frame, so
    committed flips are not delayed by the confirmation; transient excursions
    (a defender challenging, tracker noise) are cancelled instead of becoming
    spurious spells.

    At a restart the ball is AT the kicker, so a decisive vote in the first
    RESTART_SET frames with the ball within RESTART_SET_DIST of a player sets
    the team immediately (the kicking team owns the restart kick and its
    flight). The team at the kick-off of each half is read the same way.
    """
    n = len(logr_smoothed)
    labels = np.full(n, "D", dtype="<U1")
    theta = FLIP_LOGIT
    set_win = np.zeros(n, dtype=bool)
    if restart_edges is not None:
        for e in restart_edges:
            set_win[e:e + RESTART_SET] = True
    for a, b in graded_ranges:
        # initial team: first decisive vote of the half (the kick-off)
        team = HOME
        k = a
        while k < b and dead[k]:
            k += 1
        for k2 in range(k, b):
            if dead[k2]:
                break
            v = logr_smoothed[k2]
            if not np.isnan(v) and abs(v) > theta:
                team = AWAY if v < 0 else HOME
                break
        f = a
        while f < b:
            if dead[f]:
                labels[f] = "D"
                f += 1
                continue
            start = f
            while f < b and not dead[f]:
                f += 1
            stop = f
            # decode this alive span with the hold rule
            t = start
            while t < stop:
                v = logr_smoothed[t]
                if not np.isnan(v):
                    gate = bs_roll is None or np.isnan(bs_roll[t]) \
                        or bs_roll[t] < FLIP_BALL_SPEED
                    if gate and team == HOME and v < -theta:
                        if _sustained(logr_smoothed, t, stop, dead, above=False):
                            team = AWAY
                    elif gate and team == AWAY and v > theta:
                        if _sustained(logr_smoothed, t, stop, dead, above=True):
                            team = HOME
                    # restart kick: the ball is at the kicker's feet
                    if set_win[t] and d1 is not None and np.isfinite(d1[t]) \
                            and d1[t] < RESTART_SET_DIST and abs(v) > theta:
                        team = AWAY if v < 0 else HOME
                labels[t] = "H" if team == HOME else "A"
                t += 1
    return labels


def _sustained(logr, t, stop, dead, above):
    """Does the vote stay beyond +-FLIP_LOGIT (on the given side) after frame t?"""
    need = int(FLIP_HOLD * FLIP_HOLD_FRAC)
    have = 0
    seen = 0
    for k in range(t, min(stop, t + FLIP_HOLD)):
        if dead[k]:
            break
        v = logr[k]
        if np.isnan(v):
            continue
        seen += 1
        if above and v > FLIP_LOGIT:
            have += 1
        elif not above and v < -FLIP_LOGIT:
            have += 1
    return have >= need


def build_submission(out_path, cache_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    team_of_slot, names = load_team_slots(meta_path)
    print("teams: H={}, A={}".format(names[0], names[1]))

    print("loading ATD ...")
    ball, players = load_atd(raw_path, source_frames("tactical", GAME))

    # cache files are keyed by game and by the parameters they depend on, so a
    # changed GAME or VOTE_SIGMA can never silently reuse stale arrays
    feats_path = os.path.join(cache_dir, "feats_{}.npz".format(GAME)) \
        if cache_dir else None
    if feats_path and os.path.isfile(feats_path):
        data = np.load(feats_path, allow_pickle=False)
        feats = {k: data[k] for k in data.files}
    else:
        feats = extract(ball, players, team_of_slot)
        if feats_path:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(feats_path, **feats)

    vote_path = os.path.join(cache_dir, "vote_{}_s{}_g{}.npz".format(
        GAME, VOTE_SIGMA, GAP_INTERP)) if cache_dir else None
    if vote_path and os.path.isfile(vote_path):
        data = np.load(vote_path, allow_pickle=False)
        logr = data["logr"].copy()
    else:
        ball_int = interpolate_ball(ball)
        _, _, logr = team_vote_from_players(ball_int, players, team_of_slot,
                                            sigma=VOTE_SIGMA)
        if vote_path:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(vote_path, logr=logr)
    tball = ball[:, :2].copy()
    ball = None

    tags = load_tags(names)
    blocks = half_video_blocks(GAME)
    graded_ranges = [(a, b + 1) for _, a, b in blocks]

    print("dead-ball state machine ...")
    dead = dead_spans(feats, tags, graded_ranges)

    print("team vote ...")
    # broadcast second opinion: where the raw per-frame votes agree and the
    # tactical vote is undecided, add the broadcast vote before smoothing
    bvote = aligned_broadcast_vote(tball, cache_dir)
    agree = np.isfinite(bvote) & (np.sign(bvote) == np.sign(logr)) \
        & (np.abs(logr) < FLIP_LOGIT)
    logr = logr.astype(np.float64)
    logr[agree] = logr[agree] + bvote[agree]
    logr_s = rolling(logr, 2 * VOTE_SMOOTH + 1, "mean", min_periods=1)
    # kick-team prior at tagged restarts (first alive frame after a D run)
    alive_edges = np.flatnonzero(np.diff(dead.astype(int)) == -1) + 1
    for f in alive_edges:
        for t in tags:
            if t["code"].upper() == "SET PIECES" and t["start"] <= f < t["end"] \
                    and t["side"] >= 0:
                bias = TAG_PRIOR_STRENGTH if t["side"] == HOME else -TAG_PRIOR_STRENGTH
                logr_s[f:f + TAG_PRIOR] += bias
                break

    print("decoding teams ...")
    vis = feats["ball_vis"] > 0.5
    bs_roll = rolling(np.where(vis, feats["ball_speed"], np.nan), ROLL_WIN, "median")
    d1 = np.minimum(feats["d1_0"], feats["d1_1"])
    labels = decode_teams(logr_s, dead, graded_ranges, bs_roll=bs_roll, d1=d1,
                          restart_edges=alive_edges)

    graded = np.zeros(len(labels), dtype=bool)
    for a, b in graded_ranges:
        graded[a:b] = True
    labels[~graded] = "X"

    with open(out_path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(f, lab) for f, lab in enumerate(labels))
    print("wrote", out_path)

    # summary of the structure
    spells = [(a, b, k) for a, b, k in _segments(labels, graded_ranges) if k != "D"]
    print("predicted: {} D runs, {} possessions".format(
        sum(1 for a, b, k in _segments(labels, graded_ranges) if k == "D"), len(spells)))
    return labels, dead, logr_s


def _segments(labels, graded_ranges):
    out = []
    for a, b in graded_ranges:
        f = a
        while f < b:
            start = f
            lab = labels[f]
            while f < b and labels[f] == lab:
                f += 1
            out.append((start, f, lab))
    return out


if __name__ == "__main__":
    out = os.path.join(ROOT, "evaluation", "results", "possession_state_machine.csv")
    build_submission(out, cache_dir=os.path.join(HERE, "cache"))
