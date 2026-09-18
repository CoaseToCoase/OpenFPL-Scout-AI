import pandas as pd
import pytest

from experiments.arm_features import ARMS, augment, feature_lists


def _prepared():
    return pd.DataFrame({
        "_season": ["2023-24", "2023-24", "2023-24"],
        "id": [5, 6, 7],
        "gameweek": [1, 1, 1],
        "element_type": ["MID", "DEF", "FWD"],
        "web_name": ["A", "B", "C"],
        "expected_points": [1.0, 1.0, 1.0],
    })


def _ep_next():
    return pd.DataFrame({
        "season": ["2023-24", "2023-24"],
        "element_id": [5, 6],
        "gw": [1, 1],
        "ep_next": [6.5, 2.0],
        "player_code": [111, 222],
    })


def _priors():
    return pd.DataFrame({
        "season": ["2022-23"],
        "player_code": [111],
        "appearances": [30],
        "minutes": [2700],
        "points": [150],
    })


def test_ep_next_is_joined_point_in_time_not_rolled():
    out = augment(_prepared(), _ep_next(), _priors())
    assert out.loc[out["id"] == 5, "ep_next_pit"].iloc[0] == 6.5


def test_row_count_is_never_changed_by_the_join():
    out = augment(_prepared(), _ep_next(), _priors())
    assert len(out) == 3


def test_missing_ep_next_is_flagged_not_silently_zero():
    """'FPL expects nothing' and 'we have no snapshot' are different states."""
    out = augment(_prepared(), _ep_next(), _priors())
    row = out.loc[out["id"] == 7].iloc[0]
    assert row["ep_next_pit_missing"] == 1
    assert row["ep_next_pit"] == pytest.approx(4.25)  # median of 6.5 and 2.0


def test_priors_come_from_the_previous_season():
    out = augment(_prepared(), _ep_next(), _priors())
    row = out.loc[out["id"] == 5].iloc[0]
    assert row["prior_season_ppg"] == pytest.approx(5.0)       # 150 / 30
    assert row["prior_season_appearances"] == 30
    assert row["no_prior_season"] == 0


def test_player_with_no_prior_season_is_flagged():
    out = augment(_prepared(), _ep_next(), _priors())
    assert out.loc[out["id"] == 6, "no_prior_season"].iloc[0] == 1


def test_baseline_arm_keeps_web_name_and_excludes_ep_next_pit():
    cat, num = feature_lists("A_baseline")
    assert "web_name" in cat
    assert "ep_next_pit" not in num
    assert "expected_points" in num


def test_ep_next_arm_swaps_rolling_expected_points_for_point_in_time():
    cat, num = feature_lists("B_ep_next")
    assert "ep_next_pit" in num
    assert "expected_points" not in num
    assert "web_name" in cat


def test_priors_arm_drops_web_name_and_adds_priors():
    cat, num = feature_lists("C_priors")
    assert "web_name" not in cat
    assert "prior_season_ppg" in num
    assert "expected_points" in num


def test_both_arm_applies_both_changes():
    cat, num = feature_lists("D_both")
    assert "web_name" not in cat
    assert "ep_next_pit" in num
    assert "expected_points" not in num


def test_arms_are_exactly_the_four_specified():
    assert sorted(ARMS) == ["A_baseline", "B_ep_next", "C_priors", "D_both"]


def test_unknown_arm_raises():
    with pytest.raises(KeyError):
        feature_lists("E_nonsense")
