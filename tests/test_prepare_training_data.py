import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from scripts.prepare_training_data import POSITION_TO_ELEMENT_TYPE, load_season
from src.features import HISTORY_FEATURES

TEAMS_CSV = "id,name\n1,Arsenal\n2,Chelsea\n"

GW_HEADER = (
    "element,name,team,position,round,total_points,value,selected,"
    "opponent_team,was_home,clean_sheets,goals_scored\n"
)


def gw_row(element, name, position, round_, opponent_team, was_home, selected):
    return (
        f"{element},{name},Arsenal,{position},{round_},5,55,{selected},"
        f"{opponent_team},{was_home},1,1\n"
    )


class LoadSeasonTests(unittest.TestCase):
    def _write_season(self, base: Path, season: str, gw_csv: str) -> None:
        season_dir = base / season
        (season_dir / "gws").mkdir(parents=True)
        (season_dir / "teams.csv").write_text(TEAMS_CSV)
        (season_dir / "gws" / "merged_gw.csv").write_text(gw_csv)

    def test_maps_position_and_opponent_and_keeps_expected_columns(self):
        gw_csv = GW_HEADER + gw_row(101, "Saka", "MID", 1, 2, "True", 500)
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_season(base, "2023-24", gw_csv)
            teams = pd.read_csv(base / "2023-24" / "teams.csv")
            with patch("scripts.prepare_training_data.VAASTAV_DIR", base):
                df = load_season("2023-24", teams)

        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["element_type"], POSITION_TO_ELEMENT_TYPE["MID"])
        self.assertEqual(row["opponent_team_name"], "Chelsea")
        self.assertTrue(bool(row["was_home"]))
        self.assertEqual(row["_season"], "2023-24")
        for col in ["id", "element_type", "web_name", "team_name",
                    "opponent_team_name", "was_home", "gameweek",
                    "total_points", *HISTORY_FEATURES]:
            self.assertIn(col, df.columns)

    def test_drops_rows_with_unmapped_position(self):
        # "AM" (Mystery Chip pseudo-position) has no entry in
        # POSITION_TO_ELEMENT_TYPE and must be dropped, not crash or coerce.
        gw_csv = (
            GW_HEADER
            + gw_row(101, "Saka", "MID", 1, 2, "True", 500)
            + gw_row(102, "Ghost", "AM", 1, 2, "True", 10)
        )
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_season(base, "2023-24", gw_csv)
            teams = pd.read_csv(base / "2023-24" / "teams.csv")
            with patch("scripts.prepare_training_data.VAASTAV_DIR", base):
                df = load_season("2023-24", teams)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["id"], 101)

    def test_selected_by_percent_rescaled_within_gameweek(self):
        gw_csv = (
            GW_HEADER
            + gw_row(101, "Saka", "MID", 1, 2, "True", 500)
            + gw_row(102, "Rice", "MID", 1, 2, "True", 250)
        )
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_season(base, "2023-24", gw_csv)
            teams = pd.read_csv(base / "2023-24" / "teams.csv")
            with patch("scripts.prepare_training_data.VAASTAV_DIR", base):
                df = load_season("2023-24", teams)

        by_id = df.set_index("id")["selected_by_percent"]
        self.assertAlmostEqual(by_id[101], 100.0)
        self.assertAlmostEqual(by_id[102], 50.0)

    def test_missing_history_feature_columns_are_added_as_nan(self):
        gw_csv = GW_HEADER + gw_row(101, "Saka", "MID", 1, 2, "True", 500)
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_season(base, "2023-24", gw_csv)
            teams = pd.read_csv(base / "2023-24" / "teams.csv")
            with patch("scripts.prepare_training_data.VAASTAV_DIR", base):
                df = load_season("2023-24", teams)

        # expected_goals is not in our minimal GW_HEADER fixture.
        self.assertIn("expected_goals", df.columns)
        self.assertTrue(df["expected_goals"].isna().all())


if __name__ == "__main__":
    unittest.main()
