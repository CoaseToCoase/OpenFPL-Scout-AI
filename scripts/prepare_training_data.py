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
# EXTEND THIS as each season completes. On 2026-09-18 the deployed model was
# found to have never seen 2025-26 -- trainer.py holds the newest season out to
# evaluate and nobody re-fit on the full corpus, so Leeds and Sunderland were
# unknown to it despite playing the whole season, along with 137 live players
# (21%). Leaving this list to go stale reproduces exactly that bug.
# Add "2026-27" once it is complete; do not add a season still in progress.
SEASONS = ("2023-24", "2024-25", "2025-26")
POSITION_TO_ELEMENT_TYPE = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}

# vaastav column -> our canonical column (src/features.py naming)
#
# "name" is deliberately NOT mapped to web_name here: vaastav's merged_gw.csv
# "name" column is the player's FULL name ("Erling Haaland"), while the
# official FPL API -- which src/scout.py and our own live inference both
# use -- returns the short web_name ("Haaland"). Training the categorical
# web_name feature on full names made it near-useless at inference: only
# 3 of 606 live web_names exactly matched a training-time category (found
# 2026-08-30, comparing GW3 live pull against the 2025-26 training slice),
# silently zeroing the OneHotEncoder's "unknown category" path for almost
# every player and erasing ~25% of the model's own reported feature
# importance (player identity). web_name is instead joined in from each
# season's players_raw.csv (id -> web_name), the same short form the live
# API serves.
RENAME = {
    "element": "id",
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


def dedupe_fixture_rows(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per (element, fixture).

    vaastav repeats some pairs. 2025-26 lists Kroupi (element 100) GW1-9 twice,
    identically -- 10 rows that trained the model on his matches twice and
    double-weighted them in every later rolling window (found 2026-09-15).
    Exact copies collapse to one. A pair whose copies DIFFER keeps the one whose
    match has a score (vaastav's postponed-fixture phantom, e.g. 2019-20 MCI v
    ARS listed blank at GW29 and played at GW39). Anything else raises rather
    than guessing which appearance is real.
    """
    if "fixture" not in df.columns:
        raise ValueError(f"{season}: merged_gw.csv has no fixture column")
    compare = [c for c in df.columns if c != "name"]
    exact = df.duplicated(subset=compare, keep="first")
    out = df[~exact]
    repeated = out.duplicated(subset=["element", "fixture"], keep=False)
    if repeated.any():
        played = out["team_h_score"].notna() if "team_h_score" in out.columns else pd.Series(False, index=out.index)
        groups = out[repeated].groupby(["element", "fixture"])
        drop = []
        for key, g in groups:
            scored = g[played.loc[g.index]]
            if len(scored) != 1:
                raise ValueError(f"{season}: ambiguous repeated rows for element/fixture {key}")
            drop.extend(i for i in g.index if i != scored.index[0])
        out = out.drop(index=drop)
    removed = len(df) - len(out)
    if removed:
        print(f"  {season}: removed {removed} repeated (element, fixture) rows")
    return out


def load_season(season: str, teams: pd.DataFrame) -> pd.DataFrame:
    path = VAASTAV_DIR / season / "gws" / "merged_gw.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = dedupe_fixture_rows(df, season)
    df = df.drop(columns=["name"], errors="ignore")
    df = df.rename(columns=RENAME)

    web_names = pd.read_csv(
        VAASTAV_DIR / season / "players_raw.csv", encoding="utf-8-sig",
        usecols=["id", "web_name"],
    )
    df = df.merge(web_names, on="id", how="left")

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
