# Point-in-time features and player priors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether point-in-time `ep_next` and continuous player priors restore the model's compressed top end, without touching anything production depends on.

**Architecture:** A new self-contained `experiments/` package. It reads the existing `combined.csv` plus two exported CSVs, builds four feature-set variants ("arms"), trains each with the trainer's existing preprocessor and hyperparameters, and scores them on the untouched 2025-26 holdout. Production code in `src/` and `scripts/` is read, never modified, and production artifacts in `models/*.pkl` are never written.

**Tech Stack:** Python 3.9, pandas, scikit-learn, XGBoost, CatBoost, joblib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-point-in-time-features-design.md`

## Global Constraints

- **Python 3.9.6** (`.venv/bin/python`). `int | None` and `dict[str, float]` annotations FAIL at runtime on 3.9 unless the module starts with `from __future__ import annotations`. Every new module must start with that line.
- **Never write to `models/*.pkl`.** The nightly production path loads those. All arm artifacts go to `models/experiments/<arm>/`.
- **Never modify `src/features.py`, `src/scout.py`, `scripts/predict_gameweek.py`, or `trainer.py`.** Feature-list variants are built in `experiments/`, by copying the imported lists, not mutating them.
- **`RANDOM_SEED = 42`**, identical across all arms.
- **Split:** train on `_season` in `{2023-24, 2024-25}`, hold out `2025-26`. Never refit on the holdout.
- **Decision metric is `rho` and top-end coverage, NOT MAE.** Optimising MAE is what produced a model that never says 8.
- **Season format differs between sources.** `combined.csv` uses `2023-24`; the vAI DB uses `2023_24`. Convert with `.str.replace('-', '_', regex=False)`.
- Run tests with `.venv/bin/python -m pytest tests/ -q` from the repo root.

---

### Task 1: Export experiment inputs and load them safely

**Files:**
- Create: `scripts/export_experiment_inputs.sh`
- Create: `experiments/__init__.py`
- Create: `experiments/data_sources.py`
- Test: `tests/test_experiment_data_sources.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `load_ep_next_pit(path: str) -> pandas.DataFrame` with columns `season, element_id, gw, ep_next, player_code` (season already converted to `2023-24` form).
  - `load_prior_season_stats(path: str) -> pandas.DataFrame` with columns `season, player_code, appearances, minutes, points` (season in `2023-24` form).
  - Both raise `ValueError` on an empty file or missing columns.

- [ ] **Step 1: Write the export script**

Create `scripts/export_experiment_inputs.sh`. The vAI database lives on MMMS and is not reachable locally, so this is run once by hand and its output committed as data.

```bash
#!/usr/bin/env bash
# One-off export of experiment inputs from the OctoFPL-vAI DB on MMMS.
# Read-only. Run from the repo root:  bash scripts/export_experiment_inputs.sh
set -euo pipefail

HOST=monaco-mac-mini-server
DB=~/OctoFPL-vAI/db/octofpl.db
OUT=data/official
mkdir -p "$OUT"

# Point-in-time ep_next. Provenance matters: combined.csv's own
# expected_points is vaastav's xP, scraped AFTER each gameweek, so using it
# unshifted would leak. fpl_ep_next_history is rebuilt from pre-deadline
# snapshots and its loader refuses any snapshot at or after its deadline.
ssh "$HOST" "sqlite3 -header -csv $DB \"
  SELECT season, element_id, gw, ep_next, player_code
  FROM fpl_ep_next_history
  WHERE season IN ('2023_24','2024_25','2025_26')
  ORDER BY season, gw, element_id\"" > "$OUT/ep_next_pit.csv"

# Prior-season aggregates, for season N-1 of each season in combined.csv.
# gw_player_stats_historical is code-keyed and has no 'starts' column, so
# appearances (minutes > 0) and total minutes carry that signal.
ssh "$HOST" "sqlite3 -header -csv $DB \"
  SELECT season, player_code,
         SUM(minutes > 0) AS appearances,
         SUM(minutes)     AS minutes,
         SUM(points)      AS points
  FROM gw_player_stats_historical
  WHERE season IN ('2022_23','2023_24','2024_25')
  GROUP BY season, player_code
  ORDER BY season, player_code\"" > "$OUT/prior_season_stats.csv"

wc -l "$OUT/ep_next_pit.csv" "$OUT/prior_season_stats.csv"
```

- [ ] **Step 2: Run the export and confirm the known row counts**

Run: `bash scripts/export_experiment_inputs.sh`

Expected: `ep_next_pit.csv` has 86,635 lines (86,634 rows + header). `prior_season_stats.csv` has 2,448 lines (778 + 865 + 804 players + header). If these differ materially, STOP and report — the upstream table has changed since 2026-09-18 and the plan's assertions below are stale.

- [ ] **Step 3: Write the failing test**

Create `tests/test_experiment_data_sources.py`:

```python
import pandas as pd
import pytest

from experiments.data_sources import load_ep_next_pit, load_prior_season_stats

EP_CSV = "season,element_id,gw,ep_next,player_code\n2023_24,5,1,4.2,12345\n"
PRIOR_CSV = "season,player_code,appearances,minutes,points\n2022_23,12345,30,2600,140\n"


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_ep_next_season_is_converted_to_combined_csv_form(tmp_path):
    df = load_ep_next_pit(_write(tmp_path, "ep.csv", EP_CSV))
    assert df["season"].tolist() == ["2023-24"]
    assert df["ep_next"].tolist() == [4.2]


def test_prior_season_season_is_converted(tmp_path):
    df = load_prior_season_stats(_write(tmp_path, "p.csv", PRIOR_CSV))
    assert df["season"].tolist() == ["2022-23"]
    assert df["appearances"].tolist() == [30]


def test_empty_ep_next_file_raises(tmp_path):
    header_only = "season,element_id,gw,ep_next,player_code\n"
    with pytest.raises(ValueError, match="no rows"):
        load_ep_next_pit(_write(tmp_path, "empty.csv", header_only))


def test_missing_column_raises(tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        load_ep_next_pit(_write(tmp_path, "bad.csv", "season,element_id\n2023_24,5\n"))


def test_ep_next_key_is_unique(tmp_path):
    dupe = EP_CSV + "2023_24,5,1,9.9,12345\n"
    with pytest.raises(ValueError, match="duplicate"):
        load_ep_next_pit(_write(tmp_path, "dupe.csv", dupe))
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_experiment_data_sources.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments'`

- [ ] **Step 5: Write the implementation**

Create `experiments/__init__.py` as an empty file. Create `experiments/data_sources.py`:

```python
"""Load the two exported CSVs the experiment needs.

Both come from the OctoFPL-vAI DB on MMMS via scripts/export_experiment_inputs.sh.
Season strings are converted to combined.csv's form ('2023_24' -> '2023-24')
here, once, so no caller has to remember which convention it is holding.
"""
from __future__ import annotations

import pandas as pd

EP_NEXT_COLUMNS = ["season", "element_id", "gw", "ep_next", "player_code"]
PRIOR_COLUMNS = ["season", "player_code", "appearances", "minutes", "points"]


def _load(path: str, required: list) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if frame.empty:
        raise ValueError(f"{path}: no rows")
    frame["season"] = frame["season"].astype(str).str.replace("_", "-", regex=False)
    return frame


def load_ep_next_pit(path: str) -> pd.DataFrame:
    """Point-in-time pre-deadline ep_next, one row per (season, element, gw)."""
    frame = _load(path, EP_NEXT_COLUMNS)
    key = ["season", "element_id", "gw"]
    if frame.duplicated(key).any():
        raise ValueError(f"{path}: duplicate rows for {key}")
    return frame[EP_NEXT_COLUMNS]


def load_prior_season_stats(path: str) -> pd.DataFrame:
    """Season totals per player, used to build priors for the FOLLOWING season."""
    frame = _load(path, PRIOR_COLUMNS)
    key = ["season", "player_code"]
    if frame.duplicated(key).any():
        raise ValueError(f"{path}: duplicate rows for {key}")
    return frame[PRIOR_COLUMNS]
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_experiment_data_sources.py -q`
Expected: PASS, 5 passed

- [ ] **Step 7: Commit**

```bash
git add scripts/export_experiment_inputs.sh experiments/__init__.py \
        experiments/data_sources.py tests/test_experiment_data_sources.py \
        data/official/ep_next_pit.csv data/official/prior_season_stats.csv
git commit -m "experiments: export and load point-in-time ep_next and prior-season stats"
```

---

### Task 2: Build the arm feature frames

**Files:**
- Create: `experiments/arm_features.py`
- Test: `tests/test_experiment_arm_features.py`

**Interfaces:**
- Consumes: `load_ep_next_pit`, `load_prior_season_stats` from Task 1.
- Produces:
  - `ARMS` — a dict mapping arm name to `(use_ep_next_pit: bool, use_priors: bool)`, with keys exactly `"A_baseline"`, `"B_ep_next"`, `"C_priors"`, `"D_both"`.
  - `augment(prepared: pandas.DataFrame, ep_next: pandas.DataFrame, priors: pandas.DataFrame) -> pandas.DataFrame` — adds columns `ep_next_pit`, `ep_next_pit_missing`, `prior_season_ppg`, `prior_season_minutes_share`, `prior_season_appearances`, `no_prior_season`. Never drops or reorders rows.
  - `feature_lists(arm: str) -> tuple` returning `(categorical: list, numerical: list)` for that arm.

- [ ] **Step 1: Write the failing test**

Create `tests/test_experiment_arm_features.py`:

```python
import pandas as pd
import pytest

from experiments.arm_features import ARMS, augment, feature_lists


def _prepared():
    return pd.DataFrame({
        "_season": ["2023-24", "2023-24", "2023-24"],
        "id": [5, 6, 7],
        "gameweek": [1, 1, 1],
        "element_type": ["MID", "DEF", "FWD"],
        "web_name": ["A", "B", "C"],
        "expected_points": [1.0, 1.0, 1.0],
    })


def _ep_next():
    return pd.DataFrame({
        "season": ["2023-24", "2023-24"],
        "element_id": [5, 6],
        "gw": [1, 1],
        "ep_next": [6.5, 2.0],
        "player_code": [111, 222],
    })


def _priors():
    return pd.DataFrame({
        "season": ["2022-23"],
        "player_code": [111],
        "appearances": [30],
        "minutes": [2700],
        "points": [150],
    })


def test_ep_next_is_joined_point_in_time_not_rolled():
    out = augment(_prepared(), _ep_next(), _priors())
    assert out.loc[out["id"] == 5, "ep_next_pit"].iloc[0] == 6.5


def test_row_count_is_never_changed_by_the_join():
    out = augment(_prepared(), _ep_next(), _priors())
    assert len(out) == 3


def test_missing_ep_next_is_flagged_not_silently_zero():
    """'FPL expects nothing' and 'we have no snapshot' are different states."""
    out = augment(_prepared(), _ep_next(), _priors())
    row = out.loc[out["id"] == 7].iloc[0]
    assert row["ep_next_pit_missing"] == 1
    assert row["ep_next_pit"] == pytest.approx(4.25)  # median of 6.5 and 2.0


def test_priors_come_from_the_previous_season():
    out = augment(_prepared(), _ep_next(), _priors())
    row = out.loc[out["id"] == 5].iloc[0]
    assert row["prior_season_ppg"] == pytest.approx(5.0)       # 150 / 30
    assert row["prior_season_appearances"] == 30
    assert row["no_prior_season"] == 0


def test_player_with_no_prior_season_is_flagged():
    out = augment(_prepared(), _ep_next(), _priors())
    assert out.loc[out["id"] == 6, "no_prior_season"].iloc[0] == 1


def test_baseline_arm_keeps_web_name_and_excludes_ep_next_pit():
    cat, num = feature_lists("A_baseline")
    assert "web_name" in cat
    assert "ep_next_pit" not in num
    assert "expected_points" in num


def test_ep_next_arm_swaps_rolling_expected_points_for_point_in_time():
    cat, num = feature_lists("B_ep_next")
    assert "ep_next_pit" in num
    assert "expected_points" not in num
    assert "web_name" in cat


def test_priors_arm_drops_web_name_and_adds_priors():
    cat, num = feature_lists("C_priors")
    assert "web_name" not in cat
    assert "prior_season_ppg" in num
    assert "expected_points" in num


def test_both_arm_applies_both_changes():
    cat, num = feature_lists("D_both")
    assert "web_name" not in cat
    assert "ep_next_pit" in num
    assert "expected_points" not in num


def test_arms_are_exactly_the_four_specified():
    assert sorted(ARMS) == ["A_baseline", "B_ep_next", "C_priors", "D_both"]


def test_unknown_arm_raises():
    with pytest.raises(KeyError):
        feature_lists("E_nonsense")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_experiment_arm_features.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.arm_features'`

- [ ] **Step 3: Write the implementation**

Create `experiments/arm_features.py`:

```python
"""Feature-set variants for the four experiment arms.

src/features.py is imported and never mutated -- production loads the same
module. Each arm copies the lists and edits its copy.
"""
from __future__ import annotations

import pandas as pd

from src.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

PRIOR_COLUMNS = [
    "prior_season_ppg",
    "prior_season_minutes_share",
    "prior_season_appearances",
    "no_prior_season",
]
EP_NEXT_COLUMNS = ["ep_next_pit", "ep_next_pit_missing"]

# arm -> (use point-in-time ep_next, use player priors instead of web_name)
ARMS = {
    "A_baseline": (False, False),
    "B_ep_next": (True, False),
    "C_priors": (False, True),
    "D_both": (True, True),
}

_FULL_SEASON_MINUTES = 38 * 90


def _previous_season(season: str) -> str:
    """'2023-24' -> '2022-23'."""
    start = int(season[:4])
    return "%d-%02d" % (start - 1, (start % 100))


def augment(prepared: pd.DataFrame, ep_next: pd.DataFrame,
            priors: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time and prior-season columns. Row count is preserved."""
    out = prepared.copy()
    before = len(out)

    ep = ep_next.rename(columns={"element_id": "id", "gw": "gameweek"})
    out = out.merge(
        ep[["season", "id", "gameweek", "ep_next", "player_code"]],
        left_on=["_season", "id", "gameweek"],
        right_on=["season", "id", "gameweek"],
        how="left",
    ).drop(columns=["season"])
    if len(out) != before:
        raise ValueError(f"ep_next join changed row count: {before} -> {len(out)}")

    out["ep_next_pit_missing"] = out["ep_next"].isna().astype(int)
    # Impute with the position-group median for that gameweek, so "FPL expects
    # nothing" (a real 0.0) stays distinguishable from "no snapshot".
    grouped = out.groupby(["_season", "gameweek", "element_type"])["ep_next"]
    out["ep_next_pit"] = out["ep_next"].fillna(grouped.transform("median"))
    out["ep_next_pit"] = out["ep_next_pit"].fillna(out["ep_next"].median())
    out = out.drop(columns=["ep_next"])

    out["_prior_season"] = out["_season"].map(_previous_season)
    p = priors.rename(columns={"season": "_prior_season"})
    out = out.merge(p, on=["_prior_season", "player_code"], how="left")
    if len(out) != before:
        raise ValueError(f"priors join changed row count: {before} -> {len(out)}")

    out["no_prior_season"] = out["appearances"].isna().astype(int)
    apps = out["appearances"].fillna(0.0)
    out["prior_season_appearances"] = apps
    out["prior_season_ppg"] = (
        out["points"].fillna(0.0) / apps.where(apps > 0)
    ).fillna(0.0)
    out["prior_season_minutes_share"] = (
        out["minutes"].fillna(0.0) / _FULL_SEASON_MINUTES
    )
    return out.drop(columns=["_prior_season", "appearances", "minutes", "points",
                             "player_code"])


def feature_lists(arm: str) -> tuple:
    """(categorical, numerical) feature names for this arm."""
    use_ep_next, use_priors = ARMS[arm]
    categorical = list(CATEGORICAL_FEATURES)
    numerical = list(NUMERICAL_FEATURES)

    if use_ep_next:
        numerical.remove("expected_points")
        numerical.extend(EP_NEXT_COLUMNS)
    if use_priors:
        categorical.remove("web_name")
        numerical.extend(PRIOR_COLUMNS)
    return categorical, numerical
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_experiment_arm_features.py -q`
Expected: PASS, 11 passed

- [ ] **Step 5: Verify the real join against the whole corpus**

Run:

```bash
.venv/bin/python -c "
import pandas as pd
from src.features import add_rolling_history, normalize_fpl_columns
from experiments.data_sources import load_ep_next_pit, load_prior_season_stats
from experiments.arm_features import augment
raw = normalize_fpl_columns(pd.read_csv('data/official/combined.csv', low_memory=False))
raw['gameweek'] = pd.to_numeric(raw['gameweek'], errors='coerce')
raw = raw.dropna(subset=['gameweek','total_points'])
prep = add_rolling_history(raw, window=5, shift=1)
out = augment(prep, load_ep_next_pit('data/official/ep_next_pit.csv'),
              load_prior_season_stats('data/official/prior_season_stats.csv'))
print('rows', len(prep), '->', len(out))
print('ep_next missing %.2f%%' % (out.ep_next_pit_missing.mean()*100))
print('no prior season %.2f%%' % (out.no_prior_season.mean()*100))
"
```

Expected: row count unchanged, `ep_next missing` close to **0.47%**. If it exceeds 2%, STOP — the join key is wrong, not the data. `no prior season` will be materially higher (new signings and promoted players); that is expected and is exactly the case a `web_name` one-hot could not handle.

- [ ] **Step 6: Commit**

```bash
git add experiments/arm_features.py tests/test_experiment_arm_features.py
git commit -m "experiments: build the four arm feature frames"
```

---

### Task 3: Metrics that judge ordering, not averages

**Files:**
- Create: `experiments/metrics.py`
- Test: `tests/test_experiment_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `score(y_true, y_pred, gameweeks) -> dict` with keys `mae`, `rmse`, `rho`, `pred_p90`, `pred_max`, `actual_p90`, `actual_max`, `precision_at_10`, `n`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_experiment_metrics.py`:

```python
import numpy as np
import pytest

from experiments.metrics import score


def test_perfect_ordering_gives_rho_of_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = score(y, y * 2, np.array([1, 1, 1, 1]))
    assert out["rho"] == pytest.approx(1.0)


def test_reversed_ordering_gives_rho_of_minus_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = score(y, -y, np.array([1, 1, 1, 1]))
    assert out["rho"] == pytest.approx(-1.0)


def test_top_end_coverage_is_reported_for_both_sides():
    y = np.array([0.0, 10.0, 20.0])
    out = score(y, np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))
    assert out["pred_max"] == 3.0
    assert out["actual_max"] == 20.0


def test_precision_at_10_counts_returns_of_six_or_more():
    """12 players in one gameweek; our top 10 by prediction contain 4 returns."""
    pred = np.arange(12, 0, -1).astype(float)          # 12 down to 1
    y = np.array([6.0, 7.0, 2.0, 2.0, 8.0, 2.0, 2.0, 2.0, 2.0, 9.0, 2.0, 2.0])
    out = score(y, pred, np.ones(12))
    assert out["precision_at_10"] == pytest.approx(0.4)


def test_gameweek_with_fewer_than_ten_players_is_skipped():
    out = score(np.ones(5), np.ones(5), np.ones(5))
    assert out["precision_at_10"] is None


def test_n_reports_the_scored_row_count():
    assert score(np.ones(7), np.ones(7), np.ones(7))["n"] == 7
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_experiment_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.metrics'`

- [ ] **Step 3: Write the implementation**

Create `experiments/metrics.py`:

```python
"""Scoring for the experiment arms.

MAE and RMSE are reported for continuity with the existing accuracy record,
but the decision metrics are rho and top-end coverage. Optimising MAE is what
produced a model that never says 8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_THRESHOLD = 6.0
TOP_K = 10


def _rho(pred: np.ndarray, actual: np.ndarray) -> float:
    ranked_pred = pd.Series(pred).rank().to_numpy()
    ranked_actual = pd.Series(actual).rank().to_numpy()
    if ranked_pred.std() == 0 or ranked_actual.std() == 0:
        return float("nan")
    return float(np.corrcoef(ranked_pred, ranked_actual)[0, 1])


def _precision_at_k(pred: np.ndarray, actual: np.ndarray,
                    gameweeks: np.ndarray) -> object:
    hits = 0
    counted = 0
    for gameweek in np.unique(gameweeks):
        mask = gameweeks == gameweek
        if mask.sum() < TOP_K:
            continue
        top = np.argsort(-pred[mask])[:TOP_K]
        hits += int((actual[mask][top] >= RETURN_THRESHOLD).sum())
        counted += TOP_K
    if counted == 0:
        return None
    return hits / counted


def score(y_true, y_pred, gameweeks) -> dict:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    gameweeks = np.asarray(gameweeks)
    error = pred - actual
    return {
        "n": int(len(actual)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "rho": _rho(pred, actual),
        "pred_p90": float(np.percentile(pred, 90)),
        "pred_max": float(pred.max()),
        "actual_p90": float(np.percentile(actual, 90)),
        "actual_max": float(actual.max()),
        "precision_at_10": _precision_at_k(pred, actual, gameweeks),
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_experiment_metrics.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add experiments/metrics.py tests/test_experiment_metrics.py
git commit -m "experiments: add ordering and top-end coverage metrics"
```

---

### Task 4: Train and score the four arms

**Files:**
- Create: `experiments/run_arms.py`
- Test: `tests/test_experiment_run_arms.py`

**Interfaces:**
- Consumes: `ARMS`, `augment`, `feature_lists` (Task 2); `score` (Task 3); `load_ep_next_pit`, `load_prior_season_stats` (Task 1).
- Produces: `build_preprocessor(categorical: list, numerical: list) -> ColumnTransformer`; `run_arm(arm, train_df, holdout_df) -> dict`; a `main()` writing `docs/superpowers/results/2026-09-18-arm-results.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_experiment_run_arms.py`:

```python
from pathlib import Path

from experiments.run_arms import ARTIFACT_ROOT, build_preprocessor


def test_preprocessor_uses_the_arms_own_feature_lists():
    pre = build_preprocessor(["element_type"], ["gameweek"])
    names = [name for name, _, _ in pre.transformers]
    assert names == ["categorical", "numerical"]
    assert pre.transformers[0][2] == ["element_type"]
    assert pre.transformers[1][2] == ["gameweek"]


def test_artifacts_never_land_in_the_production_models_dir():
    """models/*.pkl is loaded by the nightly production path."""
    assert ARTIFACT_ROOT == Path("models") / "experiments"
    assert ARTIFACT_ROOT != Path("models")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_experiment_run_arms.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.run_arms'`

- [ ] **Step 3: Write the implementation**

Create `experiments/run_arms.py`:

```python
"""Train and score the four experiment arms on the 2025-26 holdout.

Mirrors trainer.py deliberately -- same hyperparameters, same seed, same
split, same median-of-4 aggregation -- so that the ONLY thing differing
between arms is the feature set. Writes nothing production reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from experiments.arm_features import ARMS, augment, feature_lists
from experiments.data_sources import load_ep_next_pit, load_prior_season_stats
from experiments.metrics import score
from src.features import add_rolling_history, normalize_fpl_columns

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "official" / "combined.csv"
EP_NEXT_PATH = REPO_ROOT / "data" / "official" / "ep_next_pit.csv"
PRIORS_PATH = REPO_ROOT / "data" / "official" / "prior_season_stats.csv"
RESULTS_PATH = (REPO_ROOT / "docs" / "superpowers" / "results"
                / "2026-09-18-arm-results.json")
ARTIFACT_ROOT = Path("models") / "experiments"

HOLDOUT_SEASON = "2025-26"
TARGET_COLUMN = "total_points"
RANDOM_SEED = 42


def build_preprocessor(categorical: list, numerical: list) -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical),
        ("numerical",
         Pipeline([
             ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
             ("scale", StandardScaler()),
         ]),
         numerical),
    ])


def make_models() -> dict:
    """The disclosed hyperparameters, identical to trainer.py."""
    return {
        "linear_regression": Ridge(alpha=25.0, random_state=RANDOM_SEED),
        "xgboost": XGBRegressor(
            max_depth=3, n_estimators=300, learning_rate=0.03,
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "catboost": CatBoostRegressor(
            depth=5, iterations=550, learning_rate=0.03,
            random_seed=RANDOM_SEED, verbose=False,
        ),
        # alpha=1.0 (not the disclosed 0.001) and n_iter_no_change=15 are
        # trainer.py's values and must be copied EXACTLY, or arm A will not
        # reproduce the known baseline and no comparison is trustworthy.
        "mlp": MLPRegressor(
            hidden_layer_sizes=(128, 64), alpha=1.0, batch_size=256,
            max_iter=500, random_state=RANDOM_SEED,
            early_stopping=True, n_iter_no_change=15,
        ),
    }


def run_arm(arm: str, train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> dict:
    categorical, numerical = feature_lists(arm)
    features = categorical + numerical
    x_train, y_train = train_df[features], train_df[TARGET_COLUMN].astype(float)
    x_holdout = holdout_df[features]
    y_holdout = holdout_df[TARGET_COLUMN].astype(float).to_numpy()

    out_dir = REPO_ROOT / ARTIFACT_ROOT / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for name, estimator in make_models().items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(categorical, numerical)),
            ("model", estimator),
        ])
        pipeline.fit(x_train, y_train)
        joblib.dump(pipeline, out_dir / ("%s_reg.pkl" % name))
        predictions.append(np.clip(pipeline.predict(x_holdout), 0, None))

    # median-of-4, matching the deployed aggregation exactly
    ensemble = np.clip(np.median(predictions, axis=0), 0, None)
    result = score(y_holdout, ensemble, holdout_df["gameweek"].to_numpy())
    result["arm"] = arm
    result["n_features"] = len(features)
    return result


def main() -> None:
    raw = normalize_fpl_columns(pd.read_csv(DATA_PATH, low_memory=False))
    raw["gameweek"] = pd.to_numeric(raw["gameweek"], errors="coerce")
    raw = raw.dropna(subset=["gameweek", TARGET_COLUMN])
    prepared = add_rolling_history(raw, window=5, shift=1)
    prepared = augment(prepared,
                       load_ep_next_pit(str(EP_NEXT_PATH)),
                       load_prior_season_stats(str(PRIORS_PATH)))

    train_df = prepared.loc[prepared["_season"] != HOLDOUT_SEASON].reset_index(drop=True)
    holdout_df = prepared.loc[prepared["_season"] == HOLDOUT_SEASON].reset_index(drop=True)
    print("Train: %d rows  Holdout: %d rows" % (len(train_df), len(holdout_df)))

    results = []
    for arm in ARMS:
        print("\n=== %s ===" % arm)
        result = run_arm(arm, train_df, holdout_df)
        results.append(result)
        print("  rho=%.3f  pred_max=%.2f  p90=%.2f  MAE=%.3f  P@10=%s" % (
            result["rho"], result["pred_max"], result["pred_p90"],
            result["mae"], result["precision_at_10"]))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print("\nWrote %s" % RESULTS_PATH)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_experiment_run_arms.py -q`
Expected: PASS, 2 passed

- [ ] **Step 5: Confirm production artifacts are untouched before running anything**

Run: `md5 models/*.pkl > /tmp/models_before.txt && cat /tmp/models_before.txt`

Keep this output. It is checked again in Task 5.

- [ ] **Step 6: Run the full experiment**

Run: `.venv/bin/python -m experiments.run_arms`

Expected: four arms train and print a line each; `docs/superpowers/results/2026-09-18-arm-results.json` is written. `A_baseline` should land near the known baseline (rho ~0.607, MAE ~1.106, pred_max ~8.85). If `A_baseline` deviates far from those, STOP — the harness disagrees with `trainer.py` and no arm comparison is trustworthy until that is explained.

- [ ] **Step 7: Commit**

```bash
git add experiments/run_arms.py tests/test_experiment_run_arms.py \
        docs/superpowers/results/2026-09-18-arm-results.json
git commit -m "experiments: train and score the four arms on the 2025-26 holdout"
```

---

### Task 5: Verify isolation and report the finding

**Files:**
- Create: `docs/superpowers/results/2026-09-18-arm-findings.md`
- Test: manual verification steps below

**Interfaces:**
- Consumes: `docs/superpowers/results/2026-09-18-arm-results.json` (Task 4).
- Produces: the written finding, and a recommendation on whether any arm should proceed toward deployment.

- [ ] **Step 1: Verify production artifacts were never written**

Run: `md5 models/*.pkl > /tmp/models_after.txt && diff /tmp/models_before.txt /tmp/models_after.txt && echo "UNCHANGED"`

Expected: `UNCHANGED`. If they differ, the experiment has overwritten the model the nightly path loads — restore from git or regenerate before doing anything else, and report it.

- [ ] **Step 2: Verify production code was never modified**

Run: `git diff --name-only HEAD~4 -- src/ scripts/predict_gameweek.py trainer.py`

Expected: only `scripts/export_experiment_inputs.sh` (created in Task 1). If `src/features.py`, `src/scout.py`, `trainer.py` or `scripts/predict_gameweek.py` appear, a Global Constraint has been broken — revert those files.

- [ ] **Step 3: Verify the whole suite still passes**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: all tests pass, including the pre-existing ones. The experiment must not have changed any behaviour `src/` tests depend on.

- [ ] **Step 4: Write the findings document**

Create `docs/superpowers/results/2026-09-18-arm-findings.md` containing:

1. A table of all four arms: `n`, `mae`, `rmse`, `rho`, `pred_p90`, `pred_max`, `precision_at_10`, `n_features`, with `actual_p90` and `actual_max` as the reference row.
2. The verdict against the spec's success criteria, stated plainly:
   - **Confirmed** — an arm lifts `pred_max` and `pred_p90` toward actuals with `rho` flat or better.
   - **Refuted** — the top end is unmoved, which localises the defect to the production inference path, and the next probe is a column-by-column diff of `prepare_recent_player_features` against `add_rolling_history` for the same players and gameweek.
3. Attribution: compare B against A and C against A to say which change did the work, and whether D beats both or neither.
4. The deployment blocker, restated from the spec, for whichever arm wins: `ep_next` is not currently read from the live bootstrap by `src/official_fpl.py` or `scripts/predict_gameweek.py`, and the player priors need a shipped prior-season lookup table. An arm can win cleanly and still be unshippable.
5. If MAE worsens while `rho` and the top end improve, say so and recommend proceeding anyway, with the reason: MAE is not the decision metric here.
6. **The alpha caveat for arms C and D.** `trainer.py` records that MLP `alpha`
   was raised from the disclosed 0.001 to 1.0 *specifically because* the
   ~800-column `web_name` one-hot let the net memorise player identity (train
   RMSE ~0.25, holdout RMSE 3.16). Arms C and D remove that one-hot but keep
   alpha=1.0, because the spec holds hyperparameters fixed so the arms stay
   comparable. So C and D are being judged while carrying regularisation
   chosen for a feature they no longer have. If either wins, or comes close,
   an alpha re-tune on the reduced feature set is the immediate follow-up and
   the result is a floor on what those arms can do, not a ceiling.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/results/2026-09-18-arm-findings.md
git commit -m "experiments: report the arm results and recommendation"
```
