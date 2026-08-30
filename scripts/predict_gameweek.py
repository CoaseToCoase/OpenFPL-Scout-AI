"""Score the upcoming gameweek with our own official-FPL-only ensemble and
write the result as a JSON file, uploaded to GCS for OctoFPL-vAI to ingest.

Deliberately NOT run as part of the OctoFPL-vAI nightly pipeline on MMMS
(2026-08-30 decision, Rob): that pipeline has no ML-runtime precedent --
every other prediction source (Prophet, DraftFPL) is ingested as JSON
someone else computed. Adding sklearn/xgboost/catboost as unattended
production dependencies was rejected in favour of running inference here,
locally, where the env is already built and tested, and uploading only the
resulting predictions -- the exact same shape OctoFPL-vAI's
ingest/prophet.py already consumes.

Usage:
    uv run --group train python3 scripts/predict_gameweek.py [--gameweek N]
    # then upload the printed output path:
    gsutil cp data/predictions/<season>/gw<N>.json \
        gs://octosuitedatahub/openfpl-scout/<season>/gw<N>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features import (
    MODEL_FEATURES,
    ensure_feature_columns,
    prepare_recent_player_features,
)
from src.official_fpl import OfficialFPLClient

MODELS_DIR = REPO_ROOT / "models"
MODEL_NAMES = ("linear_regression", "xgboost", "catboost", "mlp")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gameweek", type=int,
        help="Gameweek to predict (defaults to the official next event)",
    )
    parser.add_argument(
        "--history-window", type=int, default=5,
        help="Rolling-history window in past gameweeks (must match training: 5)",
    )
    return parser.parse_args(argv)


def load_ensemble() -> dict[str, object]:
    missing = [n for n in MODEL_NAMES if not (MODELS_DIR / f"{n}_reg.pkl").exists()]
    if missing:
        raise SystemExit(
            f"Missing trained model(s) {missing} in {MODELS_DIR} -- run trainer.py first"
        )
    return {n: joblib.load(MODELS_DIR / f"{n}_reg.pkl") for n in MODEL_NAMES}


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    client = OfficialFPLClient()
    bootstrap = client.bootstrap()
    gameweek = int(args.gameweek or client.next_gameweek())

    first_deadline = bootstrap["events"][0]["deadline_time"]
    season_start = int(first_deadline[:4])
    season = f"{season_start}-{str(season_start + 1)[-2:]}"

    history = client.player_history(gameweek, selectable_only=False)
    features = prepare_recent_player_features(
        history, gameweek=gameweek, history_window=args.history_window
    )
    X = ensure_feature_columns(features, MODEL_FEATURES)

    models = load_ensemble()
    per_model = {}
    for name, pipeline in models.items():
        expected = getattr(pipeline, "feature_names_in_", None)
        if expected is not None and list(expected) != MODEL_FEATURES:
            raise SystemExit(f"{name}: feature_names_in_ does not match MODEL_FEATURES")
        per_model[name] = np.clip(pipeline.predict(X), 0, None)

    ensemble_pred = np.clip(np.mean(list(per_model.values()), axis=0), 0, None)

    trained_at = None
    metadata_path = MODELS_DIR / "training_metadata.json"
    if metadata_path.exists():
        trained_at = json.loads(metadata_path.read_text()).get("trained_at_utc")

    predictions = []
    for idx, row in features.reset_index(drop=True).iterrows():
        predictions.append({
            "element_id": int(row["id"]),
            "web_name": row["web_name"],
            "team_name": row["team_name"],
            "element_type": int(row["element_type"]),
            "predicted_points": round(float(ensemble_pred[idx]), 3),
            "predicted_points_by_model": {
                name: round(float(preds[idx]), 3) for name, preds in per_model.items()
            },
        })

    output = {
        "source": "openfpl-scout-fork",
        "season": season,
        "gameweek": gameweek,
        "model_trained_at_utc": trained_at,
        "predicted_at_utc": datetime.now(timezone.utc).isoformat(),
        "player_count": len(predictions),
        "predictions": predictions,
    }

    out_dir = REPO_ROOT / "data" / "predictions" / season
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gw{gameweek}.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {len(predictions)} predictions -> {out_path}")
    print(f"Upload with: gsutil cp {out_path} "
          f"gs://octosuitedatahub/openfpl-scout/{season}/gw{gameweek}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
