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
