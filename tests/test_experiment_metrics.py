import numpy as np
import pytest

from experiments.metrics import score


def test_perfect_ordering_gives_rho_of_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = score(y, y * 2, np.array([1, 1, 1, 1]))
    assert out["rho"] == pytest.approx(1.0)


def test_reversed_ordering_gives_rho_of_minus_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = score(y, -y, np.array([1, 1, 1, 1]))
    assert out["rho"] == pytest.approx(-1.0)


def test_top_end_coverage_is_reported_for_both_sides():
    y = np.array([0.0, 10.0, 20.0])
    out = score(y, np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))
    assert out["pred_max"] == 3.0
    assert out["actual_max"] == 20.0


def test_precision_at_10_counts_returns_of_six_or_more():
    """12 players in one gameweek; our top 10 by prediction contain 4 returns."""
    pred = np.arange(12, 0, -1).astype(float)          # 12 down to 1
    y = np.array([6.0, 7.0, 2.0, 2.0, 8.0, 2.0, 2.0, 2.0, 2.0, 9.0, 2.0, 2.0])
    out = score(y, pred, np.ones(12))
    assert out["precision_at_10"] == pytest.approx(0.4)


def test_gameweek_with_fewer_than_ten_players_is_skipped():
    out = score(np.ones(5), np.ones(5), np.ones(5))
    assert out["precision_at_10"] is None


def test_n_reports_the_scored_row_count():
    assert score(np.ones(7), np.ones(7), np.ones(7))["n"] == 7
