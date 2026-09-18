import pandas as pd
import pytest

from experiments.data_sources import load_ep_next_pit, load_prior_season_stats

EP_CSV = "season,element_id,gw,ep_next,player_code\n2023_24,5,1,4.2,12345\n"
PRIOR_CSV = "season,player_code,appearances,minutes,points\n2022_23,12345,30,2600,140\n"


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_ep_next_season_is_converted_to_combined_csv_form(tmp_path):
    df = load_ep_next_pit(_write(tmp_path, "ep.csv", EP_CSV))
    assert df["season"].tolist() == ["2023-24"]
    assert df["ep_next"].tolist() == [4.2]


def test_prior_season_season_is_converted(tmp_path):
    df = load_prior_season_stats(_write(tmp_path, "p.csv", PRIOR_CSV))
    assert df["season"].tolist() == ["2022-23"]
    assert df["appearances"].tolist() == [30]


def test_empty_ep_next_file_raises(tmp_path):
    header_only = "season,element_id,gw,ep_next,player_code\n"
    with pytest.raises(ValueError, match="no rows"):
        load_ep_next_pit(_write(tmp_path, "empty.csv", header_only))


def test_missing_column_raises(tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        load_ep_next_pit(_write(tmp_path, "bad.csv", "season,element_id\n2023_24,5\n"))


def test_ep_next_key_is_unique(tmp_path):
    dupe = EP_CSV + "2023_24,5,1,9.9,12345\n"
    with pytest.raises(ValueError, match="duplicate"):
        load_ep_next_pit(_write(tmp_path, "dupe.csv", dupe))
