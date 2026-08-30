"""Walk-forward backtest: replay last season gameweek-by-gameweek through the
SAME live-inference code path (prepare_recent_player_features) that
scripts/predict_gameweek.py uses, instead of trainer.py's one-shot batch
evaluation over the whole holdout at once.

Why this is a different, more honest check: trainer.py's holdout RMSE
(~2.07) is computed by building rolling-history features for the ENTIRE
2025-26 slice in one call to add_rolling_history, using vaastav's own
naming consistently on both the train and holdout sides -- so it never
exercised the live-inference feature-prep path (prepare_recent_player_features)
and never would have caught the 2026-08-30 web_name identity bug (that bug
only broke inference-time categorical matching, not the batch-holdout
evaluation, which is why the holdout numbers barely moved after the fix
was applied). Replaying the season one gameweek at a time through the
actual live-inference function is the real test of "would this have worked
if it had been running for real last season."

Usage:
    uv run --group train python3 scripts/backtest_last_season.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features import MODEL_FEATURES, ensure_feature_columns, prepare_recent_player_features
DATA_PATH = REPO_ROOT / "data" / "official" / "combined.csv"
MODELS_DIR = REPO_ROOT / "models"
MODEL_NAMES = ("linear_regression", "xgboost", "catboost", "mlp")
SEASON = "2025-26"
TARGET_COLUMN = "total_points"
HISTORY_WINDOW = 5


def main() -> None:
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    season_data = raw[raw["_season"] == SEASON].copy()
    season_data["gameweek"] = pd.to_numeric(season_data["gameweek"], errors="coerce")
    # groupby-sum, not set_index: a double gameweek gives two rows for the
    # same (id, gameweek) -- summing matches the real total for that GW,
    # same treatment src/features.py gives double-gameweek history.
    actuals = season_data.groupby(["id", "gameweek"])[TARGET_COLUMN].sum()

    models = {n: joblib.load(MODELS_DIR / f"{n}_reg.pkl") for n in MODEL_NAMES}

    gameweeks = sorted(int(g) for g in season_data["gameweek"].dropna().unique() if g >= 2)
    per_gw_results = []
    all_errors = {n: [] for n in (*MODEL_NAMES, "ensemble")}

    for gw in gameweeks:
        history = season_data[season_data["gameweek"] < gw]
        if history.empty:
            continue
        try:
            features = prepare_recent_player_features(history, gameweek=gw, history_window=HISTORY_WINDOW)
        except ValueError:
            continue
        X = ensure_feature_columns(features, MODEL_FEATURES)

        target = actuals.reindex(list(zip(features["id"], [gw] * len(features))))
        valid_mask = target.notna().to_numpy()
        if not valid_mask.any():
            continue
        y_true = target.to_numpy()[valid_mask]

        per_model_preds = {}
        for name, pipeline in models.items():
            preds = np.clip(pipeline.predict(X), 0, None)[valid_mask]
            per_model_preds[name] = preds
            err = preds - y_true
            all_errors[name].extend(err.tolist())

        ensemble_preds = np.mean(list(per_model_preds.values()), axis=0)
        ens_err = ensemble_preds - y_true
        all_errors["ensemble"].extend(ens_err.tolist())

        per_gw_results.append({
            "gw": gw,
            "n_players": int(valid_mask.sum()),
            "ensemble_rmse": float(np.sqrt(np.mean(ens_err ** 2))),
            "ensemble_mae": float(np.mean(np.abs(ens_err))),
        })

    print(f"Walk-forward backtest: {SEASON}, gameweeks {gameweeks[0]}-{gameweeks[-1]}")
    print(f"{'GW':<4}{'n':<6}{'RMSE':<8}{'MAE':<8}")
    for r in per_gw_results:
        print(f"{r['gw']:<4}{r['n_players']:<6}{r['ensemble_rmse']:<8.3f}{r['ensemble_mae']:<8.3f}")

    print("\nSeason-aggregate (pooled across all GW x player rows):")
    summary = {}
    for name, errs in all_errors.items():
        errs = np.array(errs)
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        mae = float(np.mean(np.abs(errs)))
        summary[name] = {"rmse": rmse, "mae": mae, "n": len(errs)}
        print(f"  {name:<20} RMSE={rmse:.4f} MAE={mae:.4f} (n={len(errs)})")

    out_path = MODELS_DIR / "backtest_walkforward_2025-26.json"
    out_path.write_text(json.dumps({
        "season": SEASON, "history_window": HISTORY_WINDOW,
        "per_gw": per_gw_results, "summary": summary,
    }, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
