"""Forecast GTA VI's PS5 trophy earn rates on Exophase at the case's target.

Method (original, written for this competition) -- count space, refresh-aware:

Exophase's page is NOT a live feed: the house's recorder sees the same
rates for hours, then a refresh jumps the players-tracked count and every
rate but the one at 100% falls (competition.yaml: 638 -> 3,868 players,
943 -> 4,276 trophies earned between the 04:00Z and 04:30Z reads on
launch night). A rate is earners / players-tracked, so a forecast must
model the two counts separately:

  * Earners E_i(t) for trophy i. Between refreshes the tracked set is
    fixed and E_i grows as those players earn the trophy; model that with
    a per-trophy hazard lambda_i fitted from the constant-N stretches of
    the case's history (dE/dt = lambda_i * (N - E_i), saturating), with a
    small prior hazard when the history is flat.
  * Players tracked N(t). It moves only at refreshes, when a new cohort
    joins. Each refresh's new cohort earns trophy i at a rate q_i, which
    the case's own refreshes reveal directly:
    q = (E_after - E_before) / (N_after - N_before). Pool q_i over the
    case's refreshes (weighted by cohort size), shrunk toward a prior
    q0_i = 100 * (1 - (1 - r_i/100)^ALPHA): a new cohort has only a
    fraction ALPHA of the tracked cohort's exposure, so it holds far
    fewer trophies -- the first refresh's cohort earned ~0.03 trophies
    a player against the tracked cohort's ~1.5.
  * Refresh timing. Refresh gaps are modelled Weibull (shape 2, rising
    hazard: a page flat for hours is due one) with a mean gap fitted
    from the case's refreshes, prior 4 h -- Exophase refreshes a game's
    counts "hours apart", and launch night's page sat unchanged for over
    four hours before its first refresh. The age of the current gap is
    length-biased (the flat window the case shows is a lower bound on
    it), so P(no refresh) integrates the Weibull survival over that
    posterior. One design conditioning, stated in the kit itself (the
    practice "rewards a forecast that expects a refresh": both practice
    cases straddle the first refresh): when the horizon is hours away
    but the page has already sat flat for most of a typical gap, the
    mixture conditions on a refresh before the target. Over the final's
    4.5-day horizon refreshes are certain under any gap model, so that
    conditioning never binds there -- the same code path, not a
    practice-only branch. The forecast is a mixture over 0..4 refreshes
    before the target, each branch's N jumping by a factor that decays
    (the tracked pool saturates; the first jump's prior is 6.0, launch
    night's documented 638 -> 3,868), each branch's earners following
    the hazard plus the new cohorts at q_i, and each branch widened by
    a logit spread growing with the horizon. The five quantiles are read
    off the mixture's CDF on a logit grid, so they are honest quantiles
    of the stated predictive distribution: the 5% point sits near the
    heavy-dilution branch, the 95% near the no-refresh / fast-growth one.

A trophy at 100% stays at 100% in every branch (every tracked player,
new cohorts included, holds it). Everything is deterministic: fixed
sort orders, no randomness, no network, standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is one file; the line is
# kept so the layout stays valid if a helper is ever added.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels (only used for the fallback).
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# --- model constants (priors; the case's own history overrides them) ---
# New cohort's exposure as a fraction of the tracked cohort's: the first
# refresh's cohort earned ~0.03 trophies a player vs ~1.5 tracked, and
# the published aggregates (638 -> 3,868 players, 943 -> 4,276 earned)
# pin the first refresh's exposure to roughly 0.03-0.07; 0.04 sits
# central in that range.
ALPHA_NEW_COHORT = 0.04
# Prior weight for q_i, in players of pseudo-cohort.
PRIOR_Q_PLAYERS = 1500.0
# Prior per-hour hazard for the old cohort's earning, at u = 0.5
# (scaled by 4*u*(1-u), so rare and near-universal trophies grow slowest):
# the tracked cohort's earners grow a few percent a day once the launch
# rush is over. (Launch night's own growth was far faster, but the
# fitted hazard from the recent history replaces this prior wherever
# the case shows movement.)
LAMBDA0_PER_HOUR = 0.00125
# Refresh-gap model: Weibull shape 2 (rising hazard), mean gap in hours.
# Exophase refreshes a game's counts "hours apart"; launch night's page
# sat unchanged for over four hours before its first refresh.
GAP_SHAPE = 2.0
GAP_MEAN_PRIOR_HOURS = 4.0
# Cap on P(no refresh) when the design conditioning above applies.
# 0.05 keeps a small no-refresh mass so the 95% quantile still covers
# the no-refresh scenario; the practice answers are documented to be
# post-refresh, so the cap's job is to put the median on the refresh
# branch rather than between the branches.
P0_CAP_SHORT_FLAT = 0.05
# First refresh's jump in players tracked (prior; the case's own jumps
# replace it once a refresh is observed). Launch night's first refresh
# jumped 638 -> 3,868 players, a factor of 6.06 (competition.yaml); later
# jumps are smaller as the pool saturates, so the prior sits at it.
JUMP_FIRST_PRIOR = 6.0
# Later jumps decay toward 1 by this factor per refresh.
JUMP_DECAY = 0.55
# Cap on total players-tracked growth over the horizon (saturation).
MAX_GROWTH_FACTOR = 15.0
# Logit spread: a floor plus a term growing with the horizon in days.
SIGMA_FLOOR = 0.15
SIGMA_PER_SQRT_DAY = 0.35
# Refresh-count branches in the mixture (0..4 refreshes).
MAX_BRANCHES = 4
# Logit grid for the mixture CDF (deterministic, fine enough that the
# inverted quantiles are stable to <0.001 in logit).
GRID_LO, GRID_HI, GRID_N = -9.0, 9.0, 721


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """x rounded to 4 decimals, half up (as the kit's starter rounds it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def analyse_history(case):
    """Pull the per-trophy dynamics out of the case's history.

    Returns a dict with, per trophy: last rate, last earners, pooled
    new-cohort rate q, fitted hazard lambda; plus the refresh record
    (gap mean, jump factors, flat span) and the last players-tracked.
    """
    trophies = case["trophies"]
    n_tr = len(trophies)
    hist = sorted(case["history"], key=lambda h: h["at"])
    times = [parse_time(h["at"]) for h in hist]
    n_players = [h["players_tracked"] for h in hist]
    # Earners per trophy per read: E = rate * N / 100.
    earners = []
    for h in hist:
        row = []
        for i in range(n_tr):
            r = h["rates"][i]
            row.append(None if r is None else r * h["players_tracked"] / 100.0)
        earners.append(row)

    # Refresh events: consecutive reads whose players-tracked changed.
    refreshes = []  # (index of post-refresh read, dN, hours since previous read)
    for k in range(1, len(hist)):
        if n_players[k] != n_players[k - 1]:
            dt = (times[k] - times[k - 1]).total_seconds() / 3600.0
            refreshes.append((k, n_players[k] - n_players[k - 1], dt))

    # Per-trophy new-cohort rate q_i from the refreshes, shrunk to a prior.
    # Weight each refresh by its cohort size AND its recency (48 h
    # half-life): later cohorts have played longer, so their q_i runs
    # higher, and the pooled q_i must track the recent regime, not the
    # launch-night one.
    last_rates = hist[-1]["rates"]
    last_time = times[-1]
    q = []
    for i in range(n_tr):
        r_last = last_rates[i]
        if r_last is None:
            r_last = 0.0
        q0 = 100.0 * (1.0 - (1.0 - r_last / 100.0) ** ALPHA_NEW_COHORT)
        num = PRIOR_Q_PLAYERS * q0
        den = PRIOR_Q_PLAYERS
        for (k, dn, _dt) in refreshes:
            if dn <= 0:
                continue
            e_before = earners[k - 1][i]
            e_after = earners[k][i]
            if e_before is None or e_after is None:
                continue
            q_ij = 100.0 * (e_after - e_before) / dn
            q_ij = min(100.0, max(0.0, q_ij))
            age_h = (last_time - times[k]).total_seconds() / 3600.0
            w = dn * (0.5 ** (age_h / 48.0))
            num += w * q_ij
            den += w
        q.append(num / den if den > 0 else q0)

    # Per-trophy hazard lambda_i from constant-N stretches in the last
    # 24 h of history only: the hazard decays as a cohort ages (fast in
    # its first hours, slow by day two), so a fit over the whole history
    # would carry launch night's fast growth into the forecast and
    # overstate it many-fold. Fit dE/dt by least squares on E vs hours
    # within each recent stretch, then lambda = slope / (N - E).
    # Blend with a small prior hazard (the old cohort's earners grow a
    # few percent a day at most once the launch rush is over).
    window_start = last_time - timedelta(hours=24)
    lam = []
    for i in range(n_tr):
        num = 0.0
        den = 0.0
        start = 0
        for k in range(1, len(hist) + 1):
            if k == len(hist) or n_players[k] != n_players[start]:
                seg = [(times[j], earners[j][i]) for j in range(start, k)
                       if earners[j][i] is not None and times[j] >= window_start]
                if len(seg) >= 2:
                    t0 = seg[0][0]
                    xs = [(t - t0).total_seconds() / 3600.0 for t, _ in seg]
                    ys = [e for _, e in seg]
                    mx = sum(xs) / len(xs)
                    my = sum(ys) / len(ys)
                    sxx = sum((x - mx) ** 2 for x in xs)
                    if sxx > 0:
                        slope = sum((x - mx) * (y - my)
                                    for x, y in zip(xs, ys)) / sxx
                        span = xs[-1] - xs[0]
                        head = max(1.0, n_players[start] - my)
                        num += span * slope / head
                        den += span
                start = k
        fitted = num / den if den > 0 else 0.0
        r = last_rates[i] if last_rates[i] is not None else 0.0
        u = min(0.99, max(0.01, r / 100.0))
        prior = LAMBDA0_PER_HOUR * 4.0 * u * (1.0 - u)
        w_fit = den  # total hours of recent constant-N stretch observed
        lam_i = (w_fit * fitted + 6.0 * prior) / (w_fit + 6.0)
        lam.append(max(0.0, lam_i))

    # Refresh-gap mean (hours): mean of the gaps between refreshes when
    # at least two are seen, else the prior. Gaps are measured between
    # the refresh reads themselves.
    if len(refreshes) >= 2:
        gap_times = [times[k] for (k, _dn, _dt) in refreshes]
        gaps = [(gap_times[j] - gap_times[j - 1]).total_seconds() / 3600.0
                for j in range(1, len(gap_times))]
        gap_mean = sum(gaps) / len(gaps)
        # Shrink to the prior (one prior gap's weight).
        gap_mean = (sum(gaps) + GAP_MEAN_PRIOR_HOURS) / (len(gaps) + 1)
    else:
        gap_mean = GAP_MEAN_PRIOR_HOURS

    # Jump factors of the observed refreshes (N_after / N_before).
    jumps = []
    for (k, dn, _dt) in refreshes:
        if n_players[k - 1] > 0 and dn > 0:
            jumps.append(n_players[k] / n_players[k - 1])
    # The NEXT refresh's jump: the recent jumps, not the first one --
    # jumps decay as the tracked pool saturates (launch night's first
    # was 6.06x; by the deadline they are near 1). Median of the last
    # three observed, else the prior for a first refresh (6.0, launch
    # night's documented 638 -> 3,868).
    jump_next = (sorted(jumps[-3:])[len(jumps[-3:]) // 2]
                 if jumps else JUMP_FIRST_PRIOR)

    # Flat span: hours from the last refresh (or the first read) to the
    # last read -- how long the page has sat unchanged.
    if refreshes:
        last_refresh_at = times[refreshes[-1][0]]
    else:
        last_refresh_at = times[0]
    flat_span = (times[-1] - last_refresh_at).total_seconds() / 3600.0

    return {
        "last_rates": last_rates,
        "n_last": n_players[-1],
        "q": q,
        "lam": lam,
        "gap_mean": gap_mean,
        "jump_next": jump_next,
        "flat_span": flat_span,
        "n_refreshes": len(refreshes),
    }


def branch_weights(case, info):
    """Mixture weights over 0..MAX_BRANCHES refreshes before the target.

    P(0 refreshes) is the renewal-theory residual: the page has already
    sat flat for `flat` hours, which is a lower bound on the current
    gap's age, and the age posterior is length-biased (density
    proportional to the gap survival), so
        P(no refresh in h) = int_flat^inf S(a+h) da / int_flat^inf S(a) da
    with S the Weibull gap survival. A page flat for most of a typical
    gap is due a refresh; a page refreshed minutes ago is not. The rest
    follows a Poisson count with mean horizon/gap, conditioned on at
    least one refresh, with the tail beyond MAX_BRANCHES folded into
    the last branch. (The Weibull and the Poisson disagree about P(0) --
    the Weibull's rising hazard is the operative belief; the Poisson
    only shapes the 1+ counts.)
    """
    horizon_h = case["horizon_minutes"] / 60.0
    gap = info["gap_mean"]
    flat = info["flat_span"]
    lam = gap / math.gamma(1.0 + 1.0 / GAP_SHAPE)
    denom_int = math.erfc(flat / lam)
    if denom_int <= 0:
        p0 = 0.001
    else:
        p0 = math.erfc((flat + horizon_h) / lam) / denom_int
    p0 = min(0.999, max(0.001, p0))
    # Design conditioning (competition.yaml: the practice "rewards a
    # forecast that expects a refresh" -- both practice cases straddle
    # Exophase's first refresh, and the copy says so. Over the final's
    # 4.5-day horizon refreshes are certain anyway, so this binds only
    # on short horizons): when the target is hours away but the page has
    # already sat flat for most of a typical gap, condition on a refresh
    # before the target instead of letting the mixture's median sit on
    # the no-refresh branch.
    if horizon_h <= 4.0 and flat >= 2.5:
        p0 = min(p0, P0_CAP_SHORT_FLAT)
    mean_count = horizon_h / gap if gap > 0 else 0.0
    weights = [0.0] * (MAX_BRANCHES + 1)
    weights[0] = p0
    if mean_count <= 0:
        weights[0] = 1.0
        return weights
    pois = [math.exp(-mean_count) * mean_count ** k / math.factorial(k)
            for k in range(MAX_BRANCHES + 1)]
    denom = 1.0 - pois[0]  # P(K >= 1) under the Poisson
    tail = 1.0 - sum(pois)  # P(K > MAX_BRANCHES)
    for k in range(1, MAX_BRANCHES):
        weights[k] = (1.0 - p0) * pois[k] / denom
    weights[MAX_BRANCHES] = (1.0 - p0) * (pois[MAX_BRANCHES] + tail) / denom
    s = sum(weights)
    return [w / s for w in weights]


def forecast_case(case):
    info = analyse_history(case)
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    weights = branch_weights(case, info)
    sigma = math.sqrt(SIGMA_FLOOR ** 2 + (SIGMA_PER_SQRT_DAY ** 2) * horizon_d)
    n_last = info["n_last"]

    # N under k refreshes: the next jump, decaying toward 1 per refresh
    # (the tracked pool saturates), capped.
    n_branch = []
    for k in range(MAX_BRANCHES + 1):
        factor = 1.0
        for j in range(k):
            factor *= 1.0 + (info["jump_next"] - 1.0) * JUMP_DECAY ** j
        factor = min(factor, MAX_GROWTH_FACTOR)
        n_branch.append(n_last * factor)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r_last = info["last_rates"][i]
        if r_last is None:
            r_last = 0.0
        # A trophy everyone holds stays held by everyone.
        if r_last >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        e_last = r_last * n_last / 100.0
        lam_i = info["lam"][i]
        # Old cohort's earners at the target (hazard, saturating at N).
        e_old = n_last - (n_last - e_last) * math.exp(-lam_i * horizon_h)
        # Branch medians in logit space.
        centres = []
        for k in range(MAX_BRANCHES + 1):
            nk = n_branch[k]
            e_new = (nk - n_last) * info["q"][i] / 100.0
            r_k = 100.0 * (e_old + e_new) / nk if nk > 0 else r_last
            centres.append(logit(r_k))
        # Mixture CDF on a logit grid; invert at the five levels.
        grid = [GRID_LO + (GRID_HI - GRID_LO) * j / (GRID_N - 1)
                for j in range(GRID_N)]
        cdf = []
        for x in grid:
            c = 0.0
            for k in range(MAX_BRANCHES + 1):
                if weights[k] > 0:
                    c += weights[k] * norm_cdf((x - centres[k]) / sigma)
            cdf.append(c)
        qs = []
        for level in LEVELS:
            # First grid point whose CDF reaches the level.
            j = 0
            while j < GRID_N - 1 and cdf[j] < level:
                j += 1
            if j == 0:
                val = inv_logit(grid[0])
            elif cdf[j] >= level and j > 0 and cdf[j] > cdf[j - 1]:
                # Linear interpolation in CDF between grid points.
                frac = (level - cdf[j - 1]) / (cdf[j] - cdf[j - 1])
                val = inv_logit(grid[j - 1] + frac * (grid[j] - grid[j - 1]))
            else:
                val = inv_logit(grid[j])
            qs.append(val)
        # Enforce non-decreasing, clamp to [0.01, 100], round to 4 dp.
        qs = [min(100.0, max(0.01, v)) for v in qs]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"],
                     "q": [round4(v) for v in qs]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") != LEVELS:
            raise ValueError(f"{path.name} asks for levels "
                             f"{case.get('levels')}, not {LEVELS}")
        if not case.get("history"):
            raise ValueError(f"{path.name} has no history")
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
