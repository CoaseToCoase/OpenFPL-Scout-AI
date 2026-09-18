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


def _impute_by_position_median(raw: pd.Series, position: pd.Series) -> pd.Series:
    """Fill NaNs with the position-group median, falling back to the global
    median where a position group has no observed value at all (any NaN
    still remaining downstream is caught by the 0.0-constant SimpleImputer
    in run_arms.py, the same final fallback ep_next_pit relies on)."""
    group_median = raw.groupby(position).transform("median")
    return raw.fillna(group_median).fillna(raw.median())


def augment(prepared: pd.DataFrame, ep_next: pd.DataFrame,
            priors: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time and prior-season columns. Row count is preserved."""
    out = prepared.copy()
    before = len(out)

    # Attach player_code via season-wide mapping, not per-row join
    code_map = ep_next[["season", "element_id", "player_code"]].drop_duplicates()
    code_map = code_map.rename(columns={"element_id": "id", "season": "_season"})
    out = out.merge(code_map, on=["_season", "id"], how="left")
    if len(out) != before:
        raise ValueError(f"player_code join changed row count: {before} -> {len(out)}")

    # Get ep_next VALUE from per-row join (not player_code source)
    ep = ep_next.rename(columns={"element_id": "id", "gw": "gameweek"})
    out = out.merge(
        ep[["season", "id", "gameweek", "ep_next"]],
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
    p = priors.rename(columns={"season": "_prior_season", "minutes": "prior_minutes"})
    out = out.merge(p, on=["_prior_season", "player_code"], how="left")
    if len(out) != before:
        raise ValueError(f"priors join changed row count: {before} -> {len(out)}")

    # A player with no prior season (promoted, new signing, youth) gets the
    # position-group median rather than a hard 0.0, matching the ep_next_pit
    # imputation above -- a real zero must stay distinguishable from "no
    # data", so no_prior_season is captured BEFORE any fill.
    out["no_prior_season"] = out["appearances"].isna().astype(int)

    apps_raw = out["appearances"]
    ppg_raw = out["points"] / apps_raw.where(apps_raw > 0)
    minutes_share_raw = out["prior_minutes"] / _FULL_SEASON_MINUTES

    out["prior_season_appearances"] = _impute_by_position_median(
        apps_raw, out["element_type"])
    out["prior_season_ppg"] = _impute_by_position_median(
        ppg_raw, out["element_type"])
    out["prior_season_minutes_share"] = _impute_by_position_median(
        minutes_share_raw, out["element_type"])

    return out.drop(columns=["_prior_season", "appearances", "prior_minutes", "points",
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
