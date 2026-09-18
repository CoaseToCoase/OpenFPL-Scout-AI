import unittest

import numpy as np
import pandas as pd

from src.features import (
    CATEGORICAL_FEATURES,
    HISTORY_FEATURES,
    MODEL_FEATURES,
    add_rolling_history,
    ensure_feature_columns,
    normalize_fpl_columns,
    prepare_recent_player_features,
)

# This module was UNCOLLECTABLE from upstream e8ceadf (15 Aug 2026) until
# 2026-09-12: that commit deleted estimate_fixture_difficulty and the temporal
# feature block from src/features.py without touching the tests, so the import
# raised and pytest skipped the whole file — every test in it, including the
# ones still valid, silently stopped running for a month. Tests asserting the
# pre-refactor contract (points_mean_*, start_probability_5, fixture_difficulty,
# and "identity is not a feature") are removed rather than rewritten: those
# features no longer exist, and web_name is now deliberately IN the model as
# the identity feature.


class FeaturePreparationTests(unittest.TestCase):
    def test_model_contract_is_identity_plus_fixture_plus_rolling_history(self):
        # web_name IS a feature: the identity column was silently nullified at
        # inference until 2026-08-30 (full names vs web_name) and Haaland
        # predicted ~4 pts as a result. The fixture pair matters just as much —
        # inference must overwrite them with the UPCOMING fixture (2026-09-12).
        self.assertIn("web_name", MODEL_FEATURES)
        self.assertIn("opponent_team_name", MODEL_FEATURES)
        self.assertIn("was_home", MODEL_FEATURES)
        # ep_next_pit sits OUTSIDE HISTORY_FEATURES on purpose: it is FPL's
        # forecast for the gameweek being predicted, and rolling it over the
        # five previous gameweeks (what `expected_points` did until
        # 2026-09-18) destroys the forward signal. `expected_points` was also
        # never served by official FPL, so it was imputed to zero on every
        # production prediction while being 97% populated in training.
        self.assertEqual(
            MODEL_FEATURES,
            [*CATEGORICAL_FEATURES, "gameweek", "ep_next_pit", *HISTORY_FEATURES],
        )
        self.assertNotIn("expected_points", HISTORY_FEATURES)
        self.assertNotIn("ep_next_pit", HISTORY_FEATURES)

    def test_normalizes_legacy_stats_and_team_names(self):
        source = pd.DataFrame(
            {
                "shots": [3.0],
                "xG": [0.7],
                "Att Pen": [5.0],
                "team_name": ["Man Utd"],
                "opponent_team_name": ["Spurs"],
            }
        )

        result = normalize_fpl_columns(source)

        self.assertEqual(result.loc[0, "total_shots"], 3.0)
        self.assertEqual(result.loc[0, "expected_goals"], 0.7)
        self.assertEqual(result.loc[0, "touches_opp_box"], 5.0)
        self.assertEqual(result.loc[0, "team_name"], "Manchester United")
        self.assertEqual(result.loc[0, "opponent_team_name"], "Tottenham")
        self.assertNotIn("shots", result.columns)

    def test_rolling_history_never_uses_current_match(self):
        source = pd.DataFrame(
            {
                "_season": [2026, 2026, 2026],
                "id": [10, 10, 10],
                "web_name": ["Player", "Player", "Player"],
                "gameweek": [1, 2, 3],
                "goals": [1, 3, 8],
            }
        )

        result = add_rolling_history(source, window=2)

        self.assertTrue(np.isnan(result.loc[result.gameweek == 1, "goals"].iloc[0]))
        self.assertEqual(result.loc[result.gameweek == 2, "goals"].iloc[0], 1.0)
        self.assertEqual(result.loc[result.gameweek == 3, "goals"].iloc[0], 2.0)

    def test_double_gameweek_fixtures_share_pre_gameweek_history(self):
        source = pd.DataFrame(
            {
                "_season": [2026, 2026, 2026],
                "id": [10, 10, 10],
                "web_name": ["Player", "Player", "Player"],
                "gameweek": [1, 2, 2],
                "goals": [1, 4, 9],
            }
        )

        result = add_rolling_history(source, window=5)
        double_gameweek = result.loc[result.gameweek == 2, "goals"]

        self.assertEqual(double_gameweek.tolist(), [1.0, 1.0])

    def test_model_contract_adds_and_orders_missing_features(self):
        result = ensure_feature_columns(pd.DataFrame({"gameweek": [1]}))

        self.assertEqual(list(result.columns), MODEL_FEATURES)
        self.assertTrue(result.drop(columns="gameweek").isna().all().all())

    def test_prepares_one_player_row_from_only_prior_gameweeks(self):
        source = pd.DataFrame(
            {
                "id": [1, 1, 1, 2],
                "element_type": [3, 3, 3, 4],
                "web_name": ["One", "One", "One", "Two"],
                "team_name": ["Man Utd"] * 3 + ["Spurs"],
                "opponent_team_name": ["Arsenal"] * 4,
                "was_home": [True, False, True, False],
                "gameweek": [1, 2, 3, 1],
                "goals": [1, 3, 100, 2],
                "total_points": [1, 3, 100, 2],
            }
        )

        result = prepare_recent_player_features(source, gameweek=3, history_window=2)

        self.assertEqual(result.id.tolist(), [1, 2])
        self.assertEqual(result.loc[result.id == 1, "goals"].iloc[0], 2.0)
        self.assertEqual(result.loc[result.id == 1, "gameweek"].iloc[0], 3)
        self.assertEqual(
            result.loc[result.id == 1, "team_name"].iloc[0], "Manchester United"
        )

    def test_rejects_history_at_or_after_requested_gameweek(self):
        source = pd.DataFrame(
            {
                "id": [1],
                "element_type": [3],
                "web_name": ["One"],
                "team_name": ["Arsenal"],
                "gameweek": [2],
            }
        )

        with self.assertRaisesRegex(ValueError, "before gameweek 2"):
            prepare_recent_player_features(source, gameweek=2)


if __name__ == "__main__":
    unittest.main()


class PointInTimeEpNextTests(unittest.TestCase):
    """ep_next is FPL's forecast for the gameweek being predicted.

    Until 2026-09-18 the model used `expected_points` — a rolling mean of FPL's
    PAST forecasts — which official FPL never serves, so it was NaN and imputed
    to zero on every production prediction while being 97% populated in
    training. A top-four driver, dead at inference.
    """

    def _history(self, ep_next):
        rows = []
        for gw in (1, 2, 3):
            rows.append({
                "id": 7, "element_type": 3, "web_name": "Player",
                "team_name": "Arsenal", "opponent_team_name": "Chelsea",
                "was_home": True, "gameweek": gw, "minutes": 90,
                "total_points": 5, "ep_next": ep_next,
            })
        return pd.DataFrame(rows)

    def test_ep_next_is_taken_as_is_not_averaged(self):
        frame = prepare_recent_player_features(self._history(6.5), gameweek=4)
        self.assertEqual(frame.loc[0, "ep_next_pit"], 6.5)

    def test_ep_next_pit_is_not_a_rolled_history_feature(self):
        self.assertNotIn("ep_next_pit", HISTORY_FEATURES)
        self.assertIn("ep_next_pit", MODEL_FEATURES)

    def test_absent_ep_next_yields_nan_not_a_silent_zero(self):
        history = self._history(6.5).drop(columns=["ep_next"])
        frame = prepare_recent_player_features(history, gameweek=4)
        self.assertTrue(pd.isna(frame.loc[0, "ep_next_pit"]))
