# Point-in-time features and player priors — arm results and recommendation

**Date:** 2026-09-18
**Spec:** `docs/superpowers/specs/2026-09-18-point-in-time-features-design.md`
**Raw results:** `docs/superpowers/results/2026-09-18-arm-results.json`
**Branch:** `experiment/point-in-time-features`
**Holdout:** 2025-26, n = 29,747 player-gameweeks. Train: 2023-24 + 2024-25. Seed fixed, preprocessing shared across all four arms.

---

## Verdict

**REFUTED.**

The spec's success criterion is: *"Confirmed if an arm lifts holdout max and p90 toward actuals while rho is flat or better."*

Actual holdout max is **24.00**. The baseline reached **8.85**. Every single arm came in **lower**: B 7.65, C 6.86, D 7.79. The top end moved **away from** actuals, not toward them. p90 moved 2.77 → 2.94 at best, against an actual p90 of 4.00 — a 4% closing of a 31% gap, on the one arm that simultaneously lost 1.06 off its ceiling.

Point-in-time `ep_next` does not cause the top-end compression and does not cure it. Neither do the player priors. The hypothesis in the spec is refuted on its own stated terms.

This should not be read as a partial success, and it must not be described as the arms moving `pred_max` "closer to reality" — that reading is factually backwards against the numbers below.

---

## Results

| arm | n | MAE | RMSE | rho | pred_p90 | pred_max | P@10 | n_features |
|---|---|---|---|---|---|---|---|---|
| A_baseline | 29,747 | 1.106 | 2.071 | 0.607 | 2.765 | 8.848 | 0.300 | 38 |
| B_ep_next | 29,747 | 0.992 | 1.949 | 0.706 | 2.922 | 7.647 | **0.337** | 39 |
| C_priors | 29,747 | 1.077 | 2.064 | 0.612 | 2.726 | 6.863 | 0.326 | 41 |
| D_both | 29,747 | **0.978** | **1.944** | **0.710** | **2.942** | 7.786 | 0.324 | 42 |
| **ACTUALS** | 29,747 | — | — | — | **4.000** | **24.000** | — | — |

Bold marks the best arm per column. Note that no arm is bold on `pred_max` — the **baseline** holds the highest ceiling, and it is still 15.15 points short of the actual maximum.

---

## The separate, genuine finding: ordering skill improved substantially

This is a real result and it stands on its own. It is **not** a rescue of the refuted hypothesis.

- Spearman rho: **0.607 → 0.710**, a gain of +0.103 absolute, **+17% relative**.
- MAE fell at the same time: 1.106 → 0.978, **−12%**.
- RMSE fell too: 2.071 → 1.944, −6%.
- Precision@10 rose 0.300 → 0.324 (best is B at 0.337).

So there was **no trade-off**. Ordering improved and error shrank together. Brief item 5 ("if MAE worsens while rho improves…") does not apply: MAE did not worsen.

For a tool whose job is to rank which player to claim on waivers, **rho is the metric that matters**. A +17% relative improvement in ordering skill is worth having regardless of what the top of the distribution does. The tool ranks; it does not need to nail the absolute number to be useful, and a ranking model that is better ordered is a better ranking model.

But it is worth being precise about what this does and does not mean: a model that orders better while still refusing to predict above 8 is a model that will keep telling you the right player is at the top of the list and keep understating by how much.

---

## Attribution: `ep_next` does all the work, priors earn nothing

The four-arm design exists so a combined improvement stays attributable. It did its job cleanly.

| comparison | Δ rho | Δ MAE | Δ pred_max |
|---|---|---|---|
| B − A (ep_next alone) | **+0.099** | −0.114 | −1.201 |
| C − A (priors alone) | **+0.005** | −0.029 | **−1.985** |
| D − A (both) | +0.103 | −0.128 | −1.061 |

**B delivers +0.099 of the total +0.103 rho gain — 96% of it.** Point-in-time `ep_next` is the entire effect.

**C delivers +0.005 — essentially nothing**, well inside noise for a single-seed run. And it is not free: replacing the `web_name` one-hot with the three prior-season priors **lowers the ceiling hardest of any arm**, 8.85 → 6.86. The player priors are the worst arm on the metric the experiment was built to move.

**D beats B only marginally** (rho +0.004, MAE −0.014) and is worse than B on precision@10 (0.324 vs 0.337). Adding priors on top of `ep_next` buys close to nothing, and D carries three extra features and a shipped lookup-table dependency for it. On the evidence here, **B is the arm**: it captures nearly all the gain with strictly less deployment surface.

---

## Leakage evidence for `ep_next_pit` — verified, not just asserted

The rho gain rests entirely on `ep_next_pit` being a genuine pre-deadline snapshot. That guarantee was previously only claimed, upstream, and nothing in this branch checked it. It has now been checked empirically by joining `ep_next` at deliberately wrong gameweek offsets and correlating against actual points:

| join offset | n | rho(ep_next, actual points) |
|---|---|---|
| -2 | 81,512 | 0.727 |
| -1 | 84,265 | 0.732 |
| 0 (used) | 86,349 | 0.658 |
| +1 | 83,874 | 0.588 |
| +2 | 81,467 | 0.556 |

**rho peaks at offset -1, not at offset 0 (the offset actually used).** That is the signature of a genuine pre-deadline forecast: a snapshot taken *before* GW N's deadline inevitably encodes GW N-1's already-known result more strongly than it encodes GW N's still-unplayed one (retrodiction bleeding backward through form/fixture inputs), so correlation against last gameweek's points is higher than against this gameweek's. If the join had instead been pulling a **post-deadline** snapshot (i.e. leaking lineup/result information for GW N), offset 0 would be the peak, because a post-hoc value tracks the gameweek it actually describes best. It is not the peak here.

Two further checks point the same way:

- Among rows with `ep_next >= 2`, only **90%** actually played that gameweek. A post-lineup value — one computed after teams were announced — would show close to 99%.
- Among 0-minute rows (players who did not play), **11%** still carried `ep_next >= 1`. A post-lineup value would show close to 0%, since a value computed after the lineup is known would not predict points for a player who is not playing.

Both are consistent with a forecast made ahead of the deadline, under uncertainty about who plays, and inconsistent with a leaked post-match or post-lineup value. This is empirical support for the causal read above — B's gain is a genuine forecasting signal, not a leak.

---

## Provenance — what run this is

The two input CSVs are correctly gitignored, which means this experiment was previously pinned only by row counts. For future identification:

| file | md5 |
|---|---|
| `data/official/ep_next_pit.csv` | `0328778f2417e572dc8e220469297482` |
| `data/official/prior_season_stats.csv` | `f2ee16cc7116b0a11d11fb78cb9e206e` |
| `trainer.py` | `54ed3ddb8d57938d318c2cefb9f06fa2` |

`trainer.py` is also gitignored. Its md5 is the only guard against the hyperparameters copied into `run_arms.py` (`make_models()`) silently drifting out of sync with the real trainer — git cannot report a diff on a gitignored file, so this checksum is the sole signal that the two stayed identical for this run.

---

## The alpha caveat for C and D — honest, but it does not rescue the priors

`trainer.py` records that MLP `alpha` was raised from the disclosed 0.001 to 1.0 **specifically because** the ~800-column `web_name` one-hot let the net memorise player identity (train RMSE ~0.25 against holdout RMSE 3.16).

Arms C and D remove that one-hot but keep `alpha=1.0`, because the spec deliberately holds hyperparameters fixed so the arms stay comparable. So C and D are judged while carrying regularisation chosen for a feature they no longer have. Their numbers are a **floor on what those arms can do, not a ceiling**, and an alpha re-tune on the reduced feature set is the obvious follow-up.

That caveat is real and should be stated. It does **not** rescue the priors hypothesis, for one decisive reason: **arm B keeps the `web_name` one-hot, keeps alpha=1.0, and still captures 96% of the rho gain.** The gain is not being suppressed by mis-tuned regularisation on the priors arms — it simply is not in the priors. A re-tune might lift C and D somewhat; it cannot manufacture a signal that B already demonstrates comes from elsewhere.

---

## Two different gaps — and this experiment only addressed one

This is the most important analytical point in the document. The compression that prompted this work is **two separate compressions**, and they have been conflated.

### Gap 1 — offline ceiling: 8.85 vs actual 24.00

The model, evaluated offline on a clean holdout, never predicts above ~8.8 when reality reaches 24. This gap is **present in the baseline** and was **not fixed by any arm**; all three arms made it worse. This experiment tested the feature hypothesis for Gap 1 — that rolling-averaging point-in-time state flattens the tail — and **refuted it**. Whatever causes Gap 1, it is not `add_rolling_history` eating `expected_points`, and it is not the `web_name` one-hot's regularisation burden. The remaining candidates are structural: the loss function, the model family, the median-of-4 aggregation (measured and cleared in the spike, but only against the mean), or the simple fact that a squared-error-trained regressor on a long-tailed target will always predict near the conditional mean.

### Gap 2 — production vs offline: 5.40 vs 8.85

Production GW4 2026-27 reached only max 5.40 with rho ~0.44, while **the same artifacts** reach 8.85 / rho 0.62 offline. Same model, same gameweek number, roughly 40% worse in production.

**This experiment says nothing whatsoever about Gap 2.** Every arm here was trained and scored offline. Gap 2 was always the larger and the stranger of the two discrepancies — identical artifacts should not behave differently, and the fact that they do localises a defect to code we control and run nightly.

The spec's next probe therefore stands, unchanged and now more clearly the priority:

> a column-by-column diff of `prepare_recent_player_features` (production) against `add_rolling_history` (training), for the same players and the same gameweek.

Refutation of Gap 1's feature hypothesis is, as the spec anticipated, the more valuable outcome, precisely because it clears the ground and points at Gap 2.

---

## Deployment blocker — the winning arm is not shippable today

An arm can win cleanly and still be unshippable. This one is.

`ep_next` is the winning feature, and it is genuinely available in the live FPL bootstrap — but:

- **`src/official_fpl.py` and `scripts/predict_gameweek.py` do not currently read `ep_next` from the bootstrap.** It is present in the payload and unused. Wiring it through is new work that must happen before any deployment of arm B or D.
- **The player priors (arms C and D) additionally need a shipped prior-season lookup table.** They cannot be derived from the live bootstrap alone.
- **`MODEL_FEATURES` is validated at load time** against `feature_names_in_` (`predict_gameweek.py:108`). Changing the feature set invalidates the existing production artifacts.

Since B carries nearly all the gain and needs only the `ep_next` wiring — not the lookup table — B is the cheaper path if a deployment is pursued at all.

**Recommendation:** do not plan a deployment off this result yet. The rho gain is real and worth capturing eventually via arm B, but Gap 2 is unexplained, and shipping a retrained model into an inference path that is demonstrably 40% worse than its own offline behaviour would confound the two problems permanently. Diff the two feature paths first. Then decide.

---

## Isolation and verification

| check | method | result |
|---|---|---|
| **Step 1 — production artifacts never written** | `md5 models/*.pkl` against the pre-run snapshot | **UNCHANGED — verified by controller.** Arms were written to `models/experiments/<arm>/`; the nightly path's artifacts were never touched. |
| **Step 2 — production source never modified** | `git status src/ scripts/`, plus a separate md5 of `trainer.py` | **CLEAN — verified by controller.** `git diff HEAD~4` was deliberately not used: fix rounds added commits, and `trainer.py` is gitignored so git cannot report it modified. The md5 check covers it instead. |
| **Step 3 — full suite passes** | `.venv/bin/python -m pytest tests/ -q` | **96 passed**, 1 warning (pre-existing `NotOpenSSLWarning` from urllib3/LibreSSL, unrelated). Run by this task, including a new covering test for the prior-season imputation fix below. |

**`matmul` RuntimeWarnings.** These appeared during the arm runs. They are **pre-existing and benign** — identical warnings occur when loading the untouched production artifacts. They are not an artefact of the experiment.

---

## Summary

1. **Refuted.** No arm lifted the top end toward actuals; every arm lowered the ceiling below the baseline's 8.85 against an actual 24.00.
2. **Ordering skill improved substantially and genuinely:** rho +17% relative, with MAE down 12% at the same time — no trade-off.
3. **`ep_next` does all the work** (+0.099 of +0.103 rho); **the priors earn nothing** (+0.005) and cost the most ceiling (−1.99).
4. **C/D carry regularisation tuned for a feature they dropped** — their numbers are a floor — but B proves the gain lives in `ep_next` regardless.
5. **Gap 1 (offline ceiling) and Gap 2 (production vs offline) are different problems.** This experiment addressed only Gap 1, and refuted the feature hypothesis for it. Gap 2 is untouched and is the bigger anomaly.
6. **Next probe:** column-by-column diff of `prepare_recent_player_features` against `add_rolling_history`, same players, same gameweek.
7. **Not shippable today:** `ep_next` is in the bootstrap but unread by production code.
8. **Prior-season imputation now matches the spec.** `prior_season_ppg`, `prior_season_minutes_share` and `prior_season_appearances` are imputed with the position-group median (falling back to the global median, then the pipeline's 0.0 constant-fill) instead of a hard 0.0. Re-running confirmed A and B — which do not consume the prior columns — were bit-for-bit unchanged, so the fix did not leak into arms it shouldn't touch. C and D's numbers above are from the corrected run.
