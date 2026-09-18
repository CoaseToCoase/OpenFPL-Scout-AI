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
