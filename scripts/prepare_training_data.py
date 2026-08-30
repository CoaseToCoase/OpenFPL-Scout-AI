"""Build a combined official-FPL-only training corpus from vaastav's public
per-GW archives (github.com/vaastav/Fantasy-Premier-League).

This is our fork's substitute for the upstream trainer's data source. Upstream
trained on official FPL history PLUS a permission-pending scrape of a
third-party site ("FPL Data") for 17 extra Opta-style columns; we deliberately
exclude that source (see fork README) and leave those columns NaN, imputed at
train time exactly as src/features.py already does for any missing
season-specific statistic. vaastav's row counts for 2023-24/2024-25/2025-26
are within ~20 rows of the counts upstream's report discloses (29,725/
27,605/29,747), confirming this is effectively the same underlying data.

Output: data/official/combined.csv, one row per (season, player, gameweek)
double-gameweek fixture, with the raw (pre-rolling) columns src/features.py
expects: id, element_type, web_name, team_name, opponent_team_name, was_home,
gameweek, total_points, plus whichever HISTORY_FEATURES vaastav supplies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.features import HISTORY_FEATURES  # noqa: E402

VAASTAV_DIR = Path("/tmp/vaastav_check/data")
SEASONS = ("2023-24", "2024-25", "2025-26")
POSITION_TO_ELEMENT_TYPE = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}

# vaastav column -> our canonical column (src/features.py naming)
RENAME = {
    "element": "id",
    "name": "web_name",
    "team": "team_name",
    "round": "gameweek",
    "total_points": "total_points",
    "clean_sheets": "clean_sheet",
    "goals_scored": "goals",
    "value": "now_cost",
    "expected_goal_involvements": "expected_goal_involvements",
    "expected_goals_conceded": "expected_goals_conceded",
    "xP": "expected_points",
}


def load_season(season: str, teams: pd.DataFrame) -> pd.DataFrame:
    path = VAASTAV_DIR / season / "gws" / "merged_gw.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = df.rename(columns=RENAME)

    df["position"] = df["position"].map(POSITION_TO_ELEMENT_TYPE)
    df = df.dropna(subset=["position"])  # drops Mystery-Chip "AM" pseudo-rows
    df = df.rename(columns={"position": "element_type"})
    df["element_type"] = df["element_type"].astype(int)

    team_lookup = dict(zip(teams["id"], teams["name"]))
    df["opponent_team_name"] = df["opponent_team"].map(team_lookup)
    df["was_home"] = df["was_home"].astype(bool)

    # vaastav's per-GW files carry a raw manager COUNT ("selected"), not a
    # percentage -- rescale within each gameweek so it behaves like one
    # (relative ownership rank is what the model actually uses; the total
    # manager count itself isn't in these files, so this is a proxy, not the
    # real percentage upstream trained on).
    if "selected" in df.columns:
        max_per_gw = df.groupby("round" if "round" in df.columns else "gameweek")["selected"].transform("max")
        df["selected_by_percent"] = 100 * df["selected"] / max_per_gw.replace(0, pd.NA)

    for col in HISTORY_FEATURES:
        if col not in df.columns:
            df[col] = pd.NA

    keep = [
        "id", "element_type", "web_name", "team_name", "opponent_team_name",
        "was_home", "gameweek", "total_points", *HISTORY_FEATURES,
    ]
    df = df[keep].copy()
    df["_season"] = season  # vaastav's own label; only relative ordering matters for the CV split
    return df


def main() -> None:
    teams_by_season = {
        s: pd.read_csv(VAASTAV_DIR / s / "teams.csv", encoding="utf-8-sig")
        for s in SEASONS
    }
    frames = [load_season(s, teams_by_season[s]) for s in SEASONS]
    combined = pd.concat(frames, ignore_index=True)

    out = REPO_ROOT / "data" / "official" / "combined.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)

    print(f"Wrote {len(combined):,} rows -> {out}")
    for season, frame in zip(SEASONS, frames):
        print(f"  {season}: {len(frame):,} rows, {frame['id'].nunique()} players")
    coverage = combined[HISTORY_FEATURES].notna().mean().sort_values()
    print("\nFeature coverage (fraction non-null, excluded-source columns will show near 0):")
    for col, frac in coverage.items():
        print(f"  {col:<38} {frac:.1%}")


if __name__ == "__main__":
    main()
