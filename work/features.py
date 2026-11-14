"""Per-frame feature extraction from tactical ATD (vectorised).

Features (all metres/m·s⁻¹):
  ball_vis, ball_x, ball_y, ball_speed (|Δ|/Δt over a ±3 frame window, 25 fps)
  per team t in {0,1}: n_vis, min_d, d2, d3 (1st/2nd/3rd nearest player distance to
    ball), n5, n10 (players within 5/10 m of the ball), mean_speed, max_speed,
    mean_speed_5 (mean speed of players within 5 m)
"""
import numpy as np


def extract(ball, players, team_of_slot, pitch=(105.3, 68.0)):
    length, width = pitch
    n_frames, n_slots, _ = players.shape
    ball_vis = ~np.isnan(ball[:, 0])

    # player positions in metres, per slot (n_frames, n_slots)
    px = players[:, :, 0] * length
    py = players[:, :, 1] * width
    pv = players[:, :, 2]  # speed m/s
    p_vis = np.isfinite(px)

    bx = ball[:, 0] * length
    by = ball[:, 1] * width

    # ball speed by symmetric finite difference over 5 frames (0.2 s)
    bs = np.full(n_frames, np.nan)
    half = 2
    for dt in range(1, half + 1):
        fwd = np.full(n_frames, np.nan)
        bwd = np.full(n_frames, np.nan)
        fwd[:-dt] = np.hypot(bx[dt:] - bx[:-dt], by[dt:] - by[:-dt]) / (dt / 25.0)
        bwd[dt:] = np.hypot(bx[:-dt] - bx[dt:], by[:-dt] - by[dt:]) / (dt / 25.0)
        both = np.isfinite(fwd) & np.isfinite(bwd)
        ok = both & ~np.isfinite(bs)
        bs[ok] = (fwd[ok] + bwd[ok]) / 2.0
    # fall back to one-sided
    fwd1 = np.full(n_frames, np.nan)
    bwd1 = np.full(n_frames, np.nan)
    fwd1[:-1] = np.hypot(bx[1:] - bx[:-1], by[1:] - by[:-1]) * 25.0
    bwd1[1:] = np.hypot(bx[:-1] - bx[1:], by[:-1] - by[1:]) * 25.0
    missing = ~np.isfinite(bs)
    bs[missing] = np.nanmean([fwd1[missing], bwd1[missing]], axis=0)

    feats = {}
    feats["ball_vis"] = ball_vis.astype(np.float32)
    feats["ball_x"] = bx
    feats["ball_y"] = by
    feats["ball_speed"] = bs

    for t in (0, 1):
        mask = (team_of_slot[np.newaxis, :] == t) & p_vis  # (frames, slots)
        tx = np.where(mask, px, np.nan)
        ty = np.where(mask, py, np.nan)
        tv = np.where(mask, pv, np.nan)
        n_vis = mask.sum(axis=1).astype(np.float32)

        # distance to ball per slot; NaN propagates where either is missing
        db = np.hypot(tx - bx[:, None], ty - by[:, None])

        # 3 nearest distances
        def kth_smallest(a, k):
            out = np.full(a.shape[0], np.nan)
            tmp = np.sort(a, axis=1)
            col = np.minimum(k, tmp.shape[1] - 1)
            rows = np.arange(a.shape[0])
            out = tmp[rows, col]
            out[~np.any(np.isfinite(a), axis=1)] = np.nan
            return out

        d1 = kth_smallest(db, 0)
        d2 = kth_smallest(db, 1)
        d3 = kth_smallest(db, 2)

        n5 = (db <= 5.0).sum(axis=1).astype(np.float32)
        n10 = (db <= 10.0).sum(axis=1).astype(np.float32)

        with np.errstate(all="ignore"):
            mean_speed = np.nanmean(tv, axis=1)
            max_speed = np.nanmax(tv, axis=1)
        sp5 = np.where(db <= 5.0, tv, np.nan)
        with np.errstate(all="ignore"):
            mean_speed_5 = np.nanmean(sp5, axis=1)

        feats["n_vis_{}".format(t)] = n_vis
        feats["d1_{}".format(t)] = d1
        feats["d2_{}".format(t)] = d2
        feats["d3_{}".format(t)] = d3
        feats["n5_{}".format(t)] = n5
        feats["n10_{}".format(t)] = n10
        feats["mean_speed_{}".format(t)] = mean_speed
        feats["max_speed_{}".format(t)] = max_speed
        feats["mean_speed_5_{}".format(t)] = mean_speed_5

    return feats


def nearest_team(feats):
    """0/1 team of nearest player, -1 when no ball or no player near.
    0 = home, 1 = away."""
    d1h = feats["d1_0"]
    d1a = feats["d1_1"]
    out = np.full(len(d1h), -1, dtype=np.int8)
    ok = np.isfinite(d1h) & np.isfinite(d1a)
    out[ok] = (d1a[ok] < d1h[ok]).astype(np.int8)  # 1 when away is strictly nearest
    return out


def team_vote_from_players(ball, players, team_of_slot, sigma=3.0, pitch=(105.3, 68.0)):
    """Robust per-frame team vote: exp-weighted over all visible players.

    Returns (vote_home, vote_away) sums and log-ratio log(h/a).
    """
    length, width = pitch
    n = len(ball)
    px = players[:, :, 0] * length
    py = players[:, :, 1] * width
    bx = ball[:, 0] * length
    by = ball[:, 1] * width
    dx = px - bx[:, None]
    dy = py - by[:, None]
    d2 = dx * dx + dy * dy
    w = np.exp(-d2 / (sigma * sigma))
    w[~np.isfinite(w)] = 0.0
    vote0 = (w * (team_of_slot[None, :] == 0)).sum(axis=1)
    vote1 = (w * (team_of_slot[None, :] == 1)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logr = np.log((vote0 + 1e-9) / (vote1 + 1e-9))
    return vote0, vote1, logr
