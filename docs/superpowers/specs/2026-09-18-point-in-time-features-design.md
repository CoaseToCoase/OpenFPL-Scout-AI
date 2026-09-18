# Point-in-time features and continuous player priors

**Status:** design, not started. Experiment only — deploying any resulting model
is a separate decision (see Non-goals).
**Date:** 2026-09-18

## Problem

Our ensemble's production output is compressed at the top end. GW2-5 of 2026-27:
max 5.12-5.68, exactly 12 players projected >= 4 points, none >= 6, against
actuals reaching 17.

A spike on 2026-09-18 falsified the obvious causes:

| Suspect | Verdict | Evidence |
|---|---|---|
| median-of-4 aggregation | **not the cause** | median MAE 1.106 / rho 0.607 vs mean 1.122 / rho 0.607 on the 2025-26 holdout; the median's max is *higher* (8.85 vs 8.39) |
| an explicit upper clip | **does not exist** | `trainer.py:189` and `predict_gameweek.py:127` both clip `(0, None)` |
| early-season feature sparsity | **not the cause** | holdout GW1-4 max 8.31 vs GW10+ 7.96 |
| the 2026-09-12 opponent bug | **not the cause** | GW5 ingested 18 Sep under the fixed behaviour, still max 5.57 |

What the spike did establish: **the compression does not reproduce offline.**
Same four artifacts, same gameweek number — holdout GW4 max 8.31 / rho 0.62,
production GW4 max 5.40 / rho 0.44. The model's offline reputation is roughly
40% better than what production delivers.

Two structural causes are the leading candidates, and this experiment tests both.

### Cause 1 — point-in-time state is being rolling-averaged

Three of the 32 `HISTORY_FEATURES` are not match events. `expected_points`,
`now_cost` and `selected_by_percent` are **current state, known before the
deadline**. `add_rolling_history` replaces each row's value with the mean of the
five prior matches.

So the model never sees FPL's own forecast for the match it is predicting. It
sees the mean of FPL's forecasts for the last five. Averaging a forward forecast
across five stale values necessarily flattens the tail.

### Cause 2 — `web_name` as an ~800-column one-hot

`trainer.py:106-110` records that this one-hot forced MLP `alpha` from the
disclosed 0.001 up to 1.0. **One feature is imposing heavy regularisation on
every other feature in the model** — itself a compression mechanism. A one-hot
also cannot generalise to a player it never saw in training.

## Non-goals

- **Not deploying a model.** The output is a measurement and a recommendation.
- **Not retuning hyperparameters.** Holding them fixed is what makes the arms
  comparable. `alpha` is re-examined only if Cause 2 is confirmed.
- **Not touching the aggregation rule.** Measured, not the cause, leave it.
- **Not adding odds or nailedness.** `odds_player_fixture_probs` (101,408 rows)
  and `nailedness_snapshots` are **2026-27 only**. One season cannot train.
- **Not changing anything in OctoFPL-vAI.** This is fork-side, plus one
  read-only export.

## Design

### Data export (one-off, read-only)

From the vAI DB on MMMS to `data/official/ep_next_pit.csv`:

```sql
SELECT season, element_id, gw, ep_next, player_code
FROM fpl_ep_next_history
WHERE season IN ('2023_24','2024_25','2025_26');
```

Verified 2026-09-18: `element_id` is non-null for every row across the trainer's
three seasons (2023-24: 29,510 rows / 861 ids; 2024-25: 27,479 / 801; 2025-26:
29,645 / 840), all 38 gameweeks each. `combined.csv`'s `id` is that season's
main element id, so the join key is `(season, element_id, gameweek)` and **no
code mapping is required**.

`player_code` is exported as the bridge for prior-season priors (below): the
code comes from season N's export and the stats are read from season N-1 in
`gw_player_stats_historical`, so no extra ep_next season is needed. This is
required because
`player_season_ids` only covers 2025-26 and 2026-27 while
`gw_player_stats_historical` is code-keyed back to 2016-17.

**Provenance is the whole point of using this table.** `combined.csv`'s own
`expected_points` is vaastav's `xP`, scraped after each gameweek and documented
by vaastav as possibly containing post-match information. It is safe today only
because `add_rolling_history` shifts it. Using *that* column point-in-time would
inject leakage — the exact failure this design exists to avoid.
`fpl_ep_next_history` is rebuilt from pre-deadline snapshots and its loader
refuses any row whose snapshot is not strictly before its deadline.

### Feature changes

**`ep_next_pit`** — joined per row; added to `NUMERICAL_FEATURES`, deliberately
**not** to `HISTORY_FEATURES` so `add_rolling_history` leaves it alone.
`expected_points` is removed from `HISTORY_FEATURES` in the arms that use it.

~1% of `combined.csv` rows will have no matching snapshot. Impute with the
position-group median for that gameweek and add a `ep_next_pit_missing` boolean,
so the model can tell "FPL expects nothing" from "we have no snapshot" — these
are different states and collapsing them to 0 would be wrong.

**Player priors** — replacing `web_name` in `CATEGORICAL_FEATURES` with, joined
via `player_code` to `gw_player_stats_historical` for season N-1:

- `prior_season_ppg` — points per appearance
- `prior_season_minutes_share` — minutes / (38 * 90)
- `prior_season_appearances` — rows with `minutes > 0`

`gw_player_stats_historical` has no `starts` column, so appearances and minutes
share carry that signal instead.

A player with no prior season (promoted, new signing, youth) gets the
position-group median and a `no_prior_season` boolean. This is the case a
one-hot could never handle at all.

### Arms

Four, so that a combined improvement remains attributable:

| arm | ep_next | web_name |
|---|---|---|
| A (baseline) | rolling | one-hot |
| B | point-in-time | one-hot |
| C | rolling | priors |
| D | point-in-time | priors |

Arm A must be retrained rather than reusing the 30 Aug artifacts, so all four
share preprocessing and seed. Split is unchanged: train 2023-24 + 2024-25,
holdout 2025-26, `RANDOM_SEED` fixed.

### Metrics

Per arm, on the holdout, and per gameweek:

- **MAE, RMSE** — comparability with the existing record
- **rho** (Spearman) — ordering skill, the metric that actually matters for
  picking a player
- **p90 and max of predictions** vs actuals — the compression symptom itself
- **precision@10** — of our top 10 for a gameweek, how many returned >= 6

MAE is reported but is explicitly **not** the decision metric. Optimising MAE is
what produced a model that never says 8.

## Success criteria

The experiment answers one question: *does either change restore the top end
without costing ordering skill?*

- **Confirmed** if an arm lifts holdout max and p90 toward actuals while rho is
  flat or better. Then: assess production parity and plan a deployment.
- **Refuted** if the top end is unmoved. Then the cause is the production
  inference path itself, and the next probe is a column-by-column diff of
  `prepare_recent_player_features` against `add_rolling_history` for the same
  players and gameweek.

Either outcome is a result. Refutation is the more valuable one, because it
localises the defect to code we control and run nightly.

## Risks and constraints

**Production parity is a precondition for deployment, not for the experiment.**
Every feature must be computable pre-deadline by
`prepare_recent_player_features`:

- `ep_next_pit` — the live bootstrap carries `ep_next` per player. It is **not
  currently read** by `src/official_fpl.py` or `predict_gameweek.py`, so that is
  new work before any deployment.
- Player priors — need a shipped prior-season lookup table; they cannot be
  derived from the live bootstrap alone.

If an arm wins but its features cannot be served in production, we have learned
something real and still cannot ship it. Establish this before celebrating.

**Interface contract.** `MODEL_FEATURES` is validated at load time against
`feature_names_in_` (`predict_gameweek.py:108`). Changing it invalidates the
existing artifacts. Arms must be written to a separate `models/experiments/<arm>/`
directory and **must not overwrite `models/*.pkl`**, which the nightly
production path loads.

**Leakage.** The single largest risk. Every added column must be provably
knowable before the deadline. `ep_next_pit` qualifies by construction; prior-
season aggregates qualify because the season is over. Nothing from the current
season's future may enter, and the N-1 join must not silently pick up N.

## Open question

Whether `now_cost` and `selected_by_percent` should also become point-in-time.
Deliberately excluded from v1: they come from vaastav's `value`/`selected`,
whose provenance has the same post-gameweek uncertainty as `xP` and has not been
verified. Doing them properly needs a separate verified source. Revisit after
this result.
