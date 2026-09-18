#!/usr/bin/env python3
"""Train the deployable ensemble: point-in-time ep_next, ALL seasons.

Two deliberate differences from trainer.py, which is an EVALUATION script:

1. Trains on every season in the corpus. trainer.py holds out 2025-26 to
   measure, and nobody ever re-fit on the full data afterwards -- so the model
   in production on 2026-09-18 had never seen an entire completed season.
   Leeds and Sunderland were unknown to it despite playing all of 2025-26, as
   were 137 live players (21%).
2. Uses `ep_next_pit` (FPL's forecast for the gameweek being predicted) in
   place of `expected_points` (a rolling mean of FPL's PAST forecasts, which
   official FPL never serves, so it was imputed to zero on every production
   prediction while being 97% populated in training).

Writes to models/candidate/ -- NEVER models/ -- so a candidate is verified
against the incumbent before promotion.

    python scripts/train_production_model.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import (CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERICAL_FEATURES,
                          add_rolling_history, attach_point_in_time_ep_next,
                          normalize_fpl_columns)

RANDOM_SEED = 42


def build_preprocessor() -> ColumnTransformer:
    """Identical in structure to trainer.py's, parameterised by the current
    feature lists rather than hardcoding them."""
    return ColumnTransformer(transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         list(CATEGORICAL_FEATURES)),
        ("numerical",
         Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                   ("scale", StandardScaler())]),
         list(NUMERICAL_FEATURES)),
    ])


def make_models() -> dict:
    """trainer.py's disclosed hyperparameters, copied verbatim. trainer.py is
    gitignored, so this copy is the version-controlled record of them; if it is
    ever edited there, this will silently desync."""
    return {
        "linear_regression": Ridge(alpha=25.0, random_state=RANDOM_SEED),
        "xgboost": XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.03,
                                random_state=RANDOM_SEED, n_jobs=-1),
        "catboost": CatBoostRegressor(depth=5, iterations=550, learning_rate=0.03,
                                      random_seed=RANDOM_SEED, verbose=False),
        "mlp": MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1.0, batch_size=256,
                            max_iter=500, random_state=RANDOM_SEED,
                            early_stopping=True, n_iter_no_change=15),
    }

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "official" / "combined.csv"
EP_NEXT_PATH = REPO_ROOT / "data" / "official" / "ep_next_pit.csv"
OUT_DIR = REPO_ROOT / "models" / "candidate"
TARGET_COLUMN = "total_points"


def main() -> None:
    raw = normalize_fpl_columns(pd.read_csv(DATA_PATH, low_memory=False))
    raw["gameweek"] = pd.to_numeric(raw["gameweek"], errors="coerce")
    raw = raw.dropna(subset=["gameweek", TARGET_COLUMN])
    prepared = add_rolling_history(raw, window=5, shift=1)

    ep_next = pd.read_csv(EP_NEXT_PATH)
    ep_next["season"] = ep_next["season"].astype(str).str.replace("_", "-", regex=False)
    prepared = attach_point_in_time_ep_next(prepared, ep_next)
    missing = float(prepared["ep_next_pit"].isna().mean())
    print("rows %d, seasons %s" % (len(prepared), sorted(prepared["_season"].unique())))
    print("ep_next_pit missing: %.2f%%" % (100 * missing))
    if missing > 0.05:
        raise SystemExit("ep_next_pit missing for >5%% of rows (%.2f%%) -- check the join"
                         % (100 * missing))

    x, y = prepared[MODEL_FEATURES], prepared[TARGET_COLUMN].astype(float)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, estimator in make_models().items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", estimator),
        ])
        pipeline.fit(x, y)
        joblib.dump(pipeline, OUT_DIR / ("%s_reg.pkl" % name))
        print("  fitted", name)

    (OUT_DIR / "training_metadata.json").write_text(json.dumps({
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": sorted(prepared["_season"].unique()),
        "rows": int(len(prepared)),
        "features": MODEL_FEATURES,
        "ep_next_pit_missing_pct": round(100 * missing, 3),
    }, indent=2))
    print("\nwrote %s (candidate -- NOT promoted)" % OUT_DIR)


if __name__ == "__main__":
    main()
