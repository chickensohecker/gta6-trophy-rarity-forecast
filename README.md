# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/chickensohecker/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/chickensohecker/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every Grand Theft Auto VI PlayStation 5 trophy, the percent of
the players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z
— a week after launch — as five quantiles (5%, 25%, 50%, 75%, 95%) per
trophy, scored by mean pinball loss on the logit scale.

## How it works

`main.py` (Python 3.13, standard library only) reads one case file from
`/input` and writes one forecast file to `/output`. For each trophy it fits,
on the logit scale, a weighted least-squares line of the earn rate against
`ln(1 + hours since launch)` over the case's distinct reads (Exophase
refreshes in discrete jumps, so a frozen stretch is one observation, not
fifty), weighted toward recent reads, with Huber reweighting to shrug off
refresh glitches. It extrapolates the line to the target, keeps 60% of the
move away from the last read (a log-time line overshoots over 4+ days as the
player cohort saturates), and blends weakly toward a tier/description prior
(story chapters reach most players in a week; grinds and the platinum do
not). The spread combines a random-walk term that widens with the horizon,
the fit's own slope uncertainty extrapolated to the target, and a floor —
a 4.5-day extrapolation of a 2-day launch is genuinely uncertain, so the
band is wide on purpose.

With fewer than 3 distinct reads (the practice cases, whose histories are a
single frozen launch-night page) it falls back to persistence with the house
baseline's random-walk spread, so every case stays valid. The program is
fully deterministic: no randomness, no wall clock, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Backtest

On simulated launches (tracked-player growth + per-trophy lognormal progress
curves + Exophase-style refresh jumps, scored with the contest's own pinball
loss), this model averages ~0.43 mean pinball loss against ~0.72 for the
house persistence baseline, across varied launch regimes. The simulations are
my own model of a launch, not the real data — they guided the damping
(0.6) and the spread, they are not a promise of the final score.

---

<sub>Entered on [make.fun](https://make.fun/?ref=34622d2x&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
