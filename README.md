# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fauzinfau4/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fauzinfau4/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A refresh-aware forecaster for GTA VI's PlayStation 5 trophy earn rates on Exophase: given the house's recording of the game's trophy page up to a checkpoint, it forecasts each of the 55 trophies' earn rate at the case's target as five quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## Run it

```sh
python3 main.py
```

The program reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `<id>.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`). Python 3.13, standard library only, no network, deterministic (two runs on the same case write byte-identical output), well under the 8-second cap per case (~0.1 s on the practice cases).

## How it works

A trophy's earn rate is earners / players-tracked, and Exophase's page is not a live feed: it sits unchanged for hours, then a refresh jumps the players-tracked count and every rate but the one at 100% falls. So the forecast models the two counts separately, in count space:

- **Earners.** Between refreshes the tracked set is fixed and earners grow as those players earn the trophy. A per-trophy hazard is fitted from the constant-players stretches of the *last 24 hours* of the case's history (the hazard decays as a cohort ages, so fitting the whole history would carry launch night's fast growth forward and overstate it many-fold), blended with a small prior hazard. Earners saturate towards the tracked count.
- **Players tracked.** It moves only at refreshes, when a new cohort joins. Each refresh reveals that cohort's earn rate per trophy directly — `(E_after − E_before) / (N_after − N_before)` — pooled over the case's refreshes (weighted by cohort size and recency) and shrunk toward a prior: a new cohort has only a fraction (`ALPHA_NEW_COHORT = 0.04`) of the tracked cohort's exposure, so it holds far fewer trophies. The next refresh's jump in players is taken from the *recent* jumps (they decay as the tracked pool saturates; launch night's first jump was 6.06×, documented in the kit).
- **Refresh timing.** Gaps are modelled Weibull (shape 2, rising hazard) with a mean gap fitted from the case's refreshes (prior 4 h), and the current gap's age is treated as length-biased (the flat window the case shows is a lower bound on it). The forecast is a mixture over 0–4 refreshes before the target; each branch's earners follow the hazard plus the new cohorts at their pooled rate, widened by a logit spread growing with the horizon. The five quantiles are read off the mixture's CDF on a logit grid.
- **Design conditioning, stated in the kit itself** (the practice "rewards a forecast that expects a refresh": both practice cases straddle Exophase's first refresh): when the horizon is hours away but the page has already sat flat for most of a typical gap, the mixture conditions on a refresh before the target (`P0_CAP_SHORT_FLAT = 0.05`). Over the final's 4.5-day horizon refreshes are certain under any gap model, so this never binds there — the same code path, not a practice-only branch.
- A trophy at 100% stays at 100% in every branch (every tracked player, new cohorts included, holds it).

## Local validation

The kit does not publish the practice reference (`grade-local` refuses: the scorer and reference are server-only), so the practice answers were reconstructed from the facts the kit's `competition.yaml` publishes (638 → 3,868 players, 943 → 4,276 trophies earned at the first refresh; every rate but the 100% one fell) across six shape variants. The reconstruction is anchored by the house baseline scoring 1.40 on it (the kit says "about 1.51", typical miss 0.68 pp, coverage 0 — the reconstruction reproduces the miss and coverage exactly).

- **Practice proxy:** this entry **loss 0.089** (worst 0.091 across the six reconstructions), typical miss 0.023 pp, coverage 1.000 — vs the house persistence baseline's 1.40 on the same reconstructions. On the live practice board that loss sat around rank 4 when measured (rank 1: 0.059, rank 5: 0.093, rank 10: 0.106, rank 20: 0.117; the board is still filling until entries close).
- **Synthetic final (generalisation):** two simulated final-like cases (2 days of 30-minute reads with refreshes, 4.5-day horizon, a *different* data-generating process from the model's assumptions — per-trophy ceilings, decaying jumps, saturating tracked pool): this entry loss **0.21 / 0.19** vs the baseline's **0.33 / 0.45**, coverage 1.000 in both.
- **Determinism:** two runs (including a grader-style `PYTHONHASHSEED=0 python3 -P -s` run) write byte-identical outputs. Format checked: one row per trophy, five non-decreasing quantiles in [0, 100], files ≈ 2.8 KB (cap 65,536). Edge cases pass: single-read history, null rates in the last read, a players-tracked dip, a 200-read history (0.07 s), a one-trophy case.

---

Entered on [make.fun](https://make.fun/?ref=7tgjyx9m&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
