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
    # Drop rolling 'minutes' to avoid conflict with prior season 'minutes'
    out = out.drop(columns=["minutes"], errors="ignore")
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
