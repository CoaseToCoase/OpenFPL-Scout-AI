"""Attach the UPCOMING fixture to inference rows, and predict with it.

Shared by scripts/predict_gameweek.py and scripts/predict_season.py.

`prepare_recent_player_features` fills opponent_team_name / was_home from each
player's most recent PLAYED match, because that is the row it builds from.
Training rows carry each match's OWN opponent, so inference has to overwrite
those two fields with the fixture being predicted — otherwise the model is
asked about the wrong opponent. Measured on GW4 2026-27: 490 players move,
mean |delta| 0.13, max 0.90.

It also makes double gameweeks correct: two fixtures produce two rows for the
player and the predictions are summed. A blank gameweek yields no row at all,
which the callers render as zero.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.features import (
    MODEL_FEATURES,
    TEAM_NAME_ALIASES,
    ensure_feature_columns,
)


def team_fixtures(client) -> dict:
    """{team_name: {gw: [(opponent_name, was_home), ...]}} from the official list.

    Club names are canonicalised through TEAM_NAME_ALIASES: the fixture list
    uses the short forms ("Man City", "Spurs") and player history the long
    ones, and unmapped it silently matched no fixtures for five clubs — every
    one of their players projected 0 for the season (2026-09-12).
    """
    teams = {t["id"]: TEAM_NAME_ALIASES.get(t["name"], t["name"])
             for t in client.bootstrap()["teams"]}
    out: dict = defaultdict(lambda: defaultdict(list))
    for f in client.fixtures():
        gw = f.get("event")
        if gw is None:                      # not yet assigned to a gameweek
            continue
        home, away = teams.get(f["team_h"]), teams.get(f["team_a"])
        if not home or not away:
            continue
        out[home][int(gw)].append((away, True))
        out[away][int(gw)].append((home, False))
    return out


def predict_frame(models: dict, features: pd.DataFrame) -> np.ndarray:
    """Ensemble MEDIAN — beat the mean on MAE in 37/37 gameweeks of backtest."""
    X = ensure_feature_columns(features, MODEL_FEATURES)
    per_model = []
    for name, pipeline in models.items():
        expected = getattr(pipeline, "feature_names_in_", None)
        if expected is not None and list(expected) != MODEL_FEATURES:
            raise ValueError(f"{name}: feature_names_in_ does not match MODEL_FEATURES")
        per_model.append(np.clip(pipeline.predict(X), 0, None))
    return np.clip(np.median(per_model, axis=0), 0, None)


def predict_by_model(models: dict, features: pd.DataFrame) -> dict:
    """Per-model predictions for one already-fixture-attached frame."""
    X = ensure_feature_columns(features, MODEL_FEATURES)
    return {name: np.clip(pipeline.predict(X), 0, None) for name, pipeline in models.items()}


def attach_fixtures(base: pd.DataFrame, fixtures: dict, gw: int) -> tuple[pd.DataFrame, list]:
    """One row per (player, fixture) for `gw`. Returns (frame, element_ids)."""
    rows, index = [], []
    for _, r in base.iterrows():
        for opponent, was_home in fixtures.get(r["team_name"], {}).get(gw, []):
            row = r.copy()
            row["opponent_team_name"] = opponent
            row["was_home"] = was_home
            row["gameweek"] = gw
            rows.append(row)
            index.append(int(r["id"]))
    if not rows:
        return pd.DataFrame(columns=base.columns), []
    return pd.DataFrame(rows).reset_index(drop=True), index


def project(models: dict, base: pd.DataFrame, fixtures: dict, gws: list) -> dict:
    """{element_id: {gw: points}} — doubles summed, blanks absent."""
    out: dict = defaultdict(dict)
    for gw in gws:
        frame, index = attach_fixtures(base, fixtures, gw)
        if not index:
            continue
        preds = predict_frame(models, frame)
        for eid, p in zip(index, preds):
            out[eid][gw] = out[eid].get(gw, 0.0) + float(p)
    return out
