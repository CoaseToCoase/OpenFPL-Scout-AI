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
