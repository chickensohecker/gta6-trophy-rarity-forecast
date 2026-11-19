"""GTA VI trophy rarity forecast: predict each PS5 trophy's Exophase earn rate
at the case target, as five quantiles, from the house's reads of the trophy
page before the checkpoint.

Model (per trophy, fitted from the case's own history only):
  * Work on the logit scale, y = logit(rate), against x = ln(1 + hours since
    launch 2026-11-19T00:00:00Z). Earn rates of a launch behave roughly
    log-linearly in log time: players join, progress through chapters and
    grind at paces spread over orders of magnitude, so a trophy's rate grows
    like a CDF in log time.
  * Fit a weighted least-squares line to the trophy's distinct reads (reads
    where its rate actually changed; Exophase refreshes in discrete jumps,
    so long frozen stretches carry no information), weighted toward recent
    reads, with two rounds of Huber reweighting to shrug off refresh glitches
    (e.g. a trophy briefly reading 100% on a half-synced page).
  * Extrapolate the line to the target, then damp the move away from the last
    read (a log-time line extrapolated 4+ days overshoots as the cohort
    saturates), and blend weakly toward a tier/description prior: story
    chapters reach most players in a week, grinds and the platinum do not.
  * Spread: a random-walk term that widens with the horizon, plus the fit's
    own slope uncertainty extrapolated to the target, plus a floor. The
    pinball loss rewards an honest spread, and a 4.5-day extrapolation of a
    2-day launch is genuinely uncertain, so the band is wide on purpose.

Fallbacks keep every case valid: fewer than 3 distinct reads (the practice
cases, whose histories are one frozen page) fall back to persistence with the
house baseline's random-walk spread; a trophy with no usable reads falls back
to its prior; any unexpected failure in one trophy's fit falls back to
persistence for that trophy, never a crash.

Standard library only, deterministic: no randomness, no wall clock, inputs
processed in file order, trophies in case order.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is a single file; the line is
# kept so the layout stays valid if a helper is ever added.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# How much of the extrapolated move away from the last read we keep: a
# log-time line overshoots over a 4+ day horizon as the player cohort
# saturates, so the forecast keeps 60% of the move. Tuned on simulated
# launches (0.5-0.7 all close; 0.6 is the robust middle across fast- and
# slow-saturating regimes).
DAMP = 0.60
# Random-walk spread of a rate's logit move, per square-root day (the house
# baseline's figure; the launch's refresh jumps make it honest here too).
SIGMA_RW = 0.50
# Floor on the logit spread: even a perfect-looking fit over two days of a
# launch can be surprised by a refresh.
SPREAD_FLOOR = 0.35
# Bounds on the logit spread.
SPREAD_MIN = 0.55
SPREAD_MAX = 2.20
# How much weight the tier/description prior gets when the fit is good.
PRIOR_WEIGHT_GOOD = 0.10
# ... and when the fit is thin (few distinct reads / large residuals).
PRIOR_WEIGHT_THIN = 0.30


def parse_time(s):
    """Parse an ISO-8601 timestamp (with Z or offset) to an aware datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def hours_since_launch(when):
    """Hours from launch to `when` (an ISO string or a datetime), floored at 0."""
    if isinstance(when, str):
        when = parse_time(when)
    return max(0.0, (when - LAUNCH).total_seconds() / 3600.0)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as JavaScript Math.round does it).

    Whole numbers come back as ints so json.dumps writes 100, not 100.0.
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to ~1e-9; deterministic. Lets the program honour whatever
    quantile levels a case asks for instead of hard-coding five z values.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("norm_ppf needs 0 < p < 1")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def tier_prior_pct(trophy):
    """Weak prior for a trophy's earn rate a week after launch, in percent.

    From what the trophy asks for, not from any fitted data: story chapters
    are reached by most players who keep playing, the final mission and
    100% completion by few, grinds slower still, the platinum slowest.
    Deliberately rough -- the fitted trend carries 70-90% of the forecast.
    """
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    name = (trophy.get("name") or "").lower()
    if tier == "platinum":
        return 1.0
    if "complete chapter 1" in desc:
        return 55.0
    if "complete chapter 2" in desc:
        return 45.0
    if "complete chapter 3" in desc:
        return 35.0
    if "complete chapter 4" in desc:
        return 28.0
    if "complete the final mission" in desc and "andy" not in desc \
            and "kenton" not in desc and "billy" not in desc \
            and "randy" not in desc and "tyrone" not in desc:
        return 12.0
    if "100% completion" in desc:
        return 2.0
    if tier == "gold":
        return 3.0
    if tier == "silver":
        return 4.0
    if "complete" in desc and ("mission" in desc or "chapter" in desc):
        return 25.0
    return 6.0


def weighted_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, x_bar, sxx, n)."""
    sw = sum(ws)
    x_bar = sum(w * x for w, x in zip(ws, xs)) / sw
    y_bar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - x_bar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return y_bar, 0.0, x_bar, sxx, len(xs)
    sxy = sum(w * (x - x_bar) * (y - y_bar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return y_bar - b * x_bar, b, x_bar, sxx, len(xs)


def fit_trophy(series, x_target):
    """Fit one trophy's (x, y) series; return (median_logit, spread, ok).

    `series` is a list of (x, y, weight) with x = ln(1 + hours since
    launch), y = logit(rate). Returns ok=False when the series cannot
    support a trend fit (too few distinct points).
    """
    if len(series) < 3:
        return None, None, False
    xs = [s[0] for s in series]
    ys = [s[1] for s in series]
    ws = [s[2] for s in series]
    # Two rounds of Huber reweighting: shrug off refresh glitches.
    for _ in range(2):
        a, b, x_bar, sxx, _ = weighted_line(xs, ys, ws)
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        ordered = sorted(abs(r) for r in resid)
        mad = ordered[len(ordered) // 2] or 0.05
        scale = max(1.4826 * mad, 0.05)
        ws = [w / (1.0 + (abs(r) / (2.5 * scale)) ** 2)
              for w, r in zip(ws, resid)]
    a, b, x_bar, sxx, _ = weighted_line(xs, ys, ws)
    # Slope standard error, from the weighted residuals.
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    s2 = sum(w * r * r for w, r in zip(ws, resid)) / (sum(ws) * dof / len(xs))
    se_slope = math.sqrt(s2 / sxx) if sxx > 1e-12 else 0.0
    y_pred = a + b * x_target
    return (a, b, x_bar, se_slope, y_pred), None, True


def persistence_row(last_rate, horizon_days, levels):
    """House-baseline fallback: last rate, random-walk spread."""
    spread = SIGMA_RW * math.sqrt(max(horizon_days, 0.0))
    lg = logit(last_rate if last_rate is not None else 0.0)
    return [round4(clamp_pct(logistic(lg + norm_ppf(tau) * spread)))
            for tau in levels]


def forecast_trophy(trophy, reads, horizon_days, levels, x_target):
    """Forecast one trophy's quantiles. `reads` is the case history."""
    idx = trophy["_index"]
    last_rate = None
    for read in reversed(reads):
        rate = read["rates"][idx]
        if rate is not None:
            last_rate = rate
            break
    prior_pct = tier_prior_pct(trophy)

    # Build the trophy's series from distinct reads only: Exophase refreshes
    # in jumps, so a frozen stretch is one observation, not fifty.
    series = []
    prev_rate = None
    for read in reads:
        rate = read["rates"][idx]
        if rate is None:
            continue
        if prev_rate is not None and rate == prev_rate:
            continue
        prev_rate = rate
        t = hours_since_launch(read["at"])
        series.append((math.log1p(t), logit(rate), read["at"]))
    if len(series) >= 3:
        t_first = series[0][2]
        t_last = series[-1][2]
        span = max(1.0, (t_last - t_first).total_seconds())
        weighted = []
        for x, y, when in series:
            # Recency ramp: the oldest distinct read counts half, the
            # newest full.
            rec = 0.5 + 0.5 * (when - t_first).total_seconds() / span
            weighted.append((x, y, rec))
        fit, _, ok = fit_trophy(weighted, x_target)
        if ok:
            a, b, x_bar, se_slope, y_pred = fit
            y_last = logit(last_rate)
            y_med = y_last + DAMP * (y_pred - y_last)
            # Prior blend: more prior when the fit is thin.
            w_prior = PRIOR_WEIGHT_GOOD if len(series) >= 8 else PRIOR_WEIGHT_THIN
            y_med = (1.0 - w_prior) * y_med + w_prior * logit(prior_pct)
            # Spread: random walk + slope uncertainty extrapolated + floor.
            var = (SIGMA_RW * math.sqrt(max(horizon_days, 0.0))) ** 2
            var += (se_slope * max(0.0, x_target - x_bar)) ** 2
            var += SPREAD_FLOOR ** 2
            spread = min(SPREAD_MAX, max(SPREAD_MIN, math.sqrt(var)))
            qs = [round4(clamp_pct(logistic(y_med + norm_ppf(tau) * spread)))
                  for tau in levels]
            # Enforce non-decreasing after rounding/clamping.
            for k in range(1, len(qs)):
                if qs[k] < qs[k - 1]:
                    qs[k] = qs[k - 1]
            return qs
    if last_rate is not None:
        return persistence_row(last_rate, horizon_days, levels)
    # No usable reads at all: the prior, wide.
    lg = logit(prior_pct)
    spread = SPREAD_MAX
    return [round4(clamp_pct(logistic(lg + norm_ppf(tau) * spread)))
            for tau in levels]


def forecast_case(case, name):
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(hours_since_launch(case["target"]))
    trophies = case["trophies"]
    for i, trophy in enumerate(trophies):
        trophy["_index"] = i
    reads = []
    for read in case.get("history") or []:
        reads.append({
            "at": parse_time(read["at"]),
            "rates": read["rates"],
        })
    rows = []
    for trophy in trophies:
        try:
            qs = forecast_trophy(trophy, reads, horizon_days, levels, x_target)
        except Exception:
            last = None
            for read in reversed(reads):
                rate = read["rates"][trophy["_index"]]
                if rate is not None:
                    last = rate
                    break
            qs = persistence_row(last, horizon_days, levels)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
