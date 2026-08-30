import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from scripts.prepare_training_data import POSITION_TO_ELEMENT_TYPE, load_season
from src.features import HISTORY_FEATURES

TEAMS_CSV = "id,name\n1,Arsenal\n2,Chelsea\n"

# "name" (full name, e.g. "Bukayo Saka") deliberately differs from the
# players_raw.csv "web_name" (e.g. "Saka") in these fixtures -- that gap is
# the exact live-inference mismatch the 2026-08-30 fix targets. Tests assert
# web_name comes from players_raw.csv, never from the per-GW "name" column.
GW_HEADER = (
    "element,name,team,position,round,total_points,value,selected,"
    "opponent_team,was_home,clean_sheets,goals_scored\n"
)


def gw_row(element, full_name, position, round_, opponent_team, was_home, selected):
    return (
        f"{element},{full_name},Arsenal,{position},{round_},5,55,{selected},"
        f"{opponent_team},{was_home},1,1\n"
    )


def players_raw_row(element_id, web_name):
    return f"{element_id},{web_name}\n"


class LoadSeasonTests(unittest.TestCase):
    def _write_season(self, base: Path, season: str, gw_csv: str,
                       players_raw_rows: str = "") -> None:
        season_dir = base / season
        (season_dir / "gws").mkdir(parents=True)
        (season_dir / "teams.csv").write_text(TEAMS_CSV)
        (season_dir / "gws" / "merged_gw.csv").write_text(gw_csv)
        (season_dir / "players_raw.csv").write_text(
            "id,web_name\n" + players_raw_rows
        )

    def test_maps_position_and_opponent_and_keeps_expected_columns(self):
        gw_csv = GW_HEADER + gw_row(101, "Bukayo Saka", "MID", 1, 2, "True", 500)
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_season(base, "2023-24", gw_csv, players_raw_row(101, "Saka"))
            teams = pd.read_csv(base / "2023-24" / "teams.csv")
            with patch("scripts.prepare_training_data.VAASTAV_DIR", base):
                df = load_season("2023-24", teams)

        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["element_type"], POSITION_TO_ELEMENT_TYPE["MID"])
        self.assertEqual(row["opponent_team_name"], "Chelsea")
        self.assertTrue(bool(row["was_home"]))
        self.assertEqual(row["_season"], "2023-24")
        # short web_name from players_raw.csv, NOT the full "Bukayo Saka"
        # that vaastav's per-GW "name" column carries.
        self.assertEqual(row["web_name"], "Saka")
        for col in ["id", "element_type", "web_name", "team_name",
                    "opponent_team_name", "was_home", "gameweek",
                    "total_points", *HISTORY_FEATURES]:
            self.assertIn(col, df.columns)

    def test_drops_rows_with_unmapped_position(self):
        # "AM" (Mystery Chip pseudo-position) has no entry in
        # POSITION_TO_ELEMENT_TYPE and must be dropped, not crash or coerce.
        gw_csv = (
            GW_HEADER
            + gw_row(101, "Bukayo Saka", "MID", 1, 2, "True", 500)
            + gw_row(102, "Ghost Player", "AM", 1, 2, "True", 10)
        )
        raw = players_raw_row(101, "Saka") + players_raw_row(102, "Ghost")
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_season(base, "2023-24", gw_csv, raw)
            teams = pd.read_csv(base / "2023-24" / "teams.csv")
            with patch("scripts.prepare_training_data.VAASTAV_DIR", base):
                df = load_season("2023-24", teams)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["id"], 101)

    def test_selected_by_percent_rescaled_within_gameweek(self):
        gw_csv = (
            GW_HEADER
            + gw_row(101, "Bukayo Saka", "MID", 1, 2, "True", 500)
            + gw_row(102, "Declan Rice", "MID", 1, 2, "True", 250)
        )
        raw = players_raw_row(101, "Saka") + players_raw_row(102, "Rice")
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_season(base, "2023-24", gw_csv, raw)
            teams = pd.read_csv(base / "2023-24" / "teams.csv")
            with patch("scripts.prepare_training_data.VAASTAV_DIR", base):
                df = load_season("2023-24", teams)

        by_id = df.set_index("id")["selected_by_percent"]
        self.assertAlmostEqual(by_id[101], 100.0)
        self.assertAlmostEqual(by_id[102], 50.0)

    def test_missing_history_feature_columns_are_added_as_nan(self):
        gw_csv = GW_HEADER + gw_row(101, "Bukayo Saka", "MID", 1, 2, "True", 500)
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_season(base, "2023-24", gw_csv, players_raw_row(101, "Saka"))
            teams = pd.read_csv(base / "2023-24" / "teams.csv")
            with patch("scripts.prepare_training_data.VAASTAV_DIR", base):
                df = load_season("2023-24", teams)

        # expected_goals is not in our minimal GW_HEADER fixture.
        self.assertIn("expected_goals", df.columns)
        self.assertTrue(df["expected_goals"].isna().all())

    def test_web_name_matches_official_api_short_form_not_full_name(self):
        # Regression test for the 2026-08-30 bug: only 3 of 606 live
        # web_names matched a training-time category because vaastav's
        # per-GW "name" column ("Erling Haaland") was fed in as web_name
        # instead of the official API's short form ("Haaland"), silently
        # nullifying the identity feature for nearly every player at
        # inference time.
        gw_csv = GW_HEADER + gw_row(411, "Erling Haaland", "FWD", 1, 2, "True", 500)
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_season(base, "2023-24", gw_csv, players_raw_row(411, "Haaland"))
            teams = pd.read_csv(base / "2023-24" / "teams.csv")
            with patch("scripts.prepare_training_data.VAASTAV_DIR", base):
                df = load_season("2023-24", teams)

        self.assertEqual(df.iloc[0]["web_name"], "Haaland")
        self.assertNotEqual(df.iloc[0]["web_name"], "Erling Haaland")


if __name__ == "__main__":
    unittest.main()
