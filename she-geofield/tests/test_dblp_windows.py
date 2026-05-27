from pathlib import Path

from she_geofield.dblp.filters import filter_records
from she_geofield.dblp.parse_xml import load_dblp_records
from she_geofield.dblp.windows import build_rolling_windows


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dblp_sample.xml"


def test_build_rolling_windows_is_deterministic():
    records = filter_records(
        load_dblp_records(FIXTURE),
        start_year=2020,
        end_year=2023,
        min_team_size=2,
        max_team_size=4,
    )
    windows = build_rolling_windows(records, width=2, stride=1, start_year=2020, end_year=2023)

    assert [window.label for window in windows] == ["2020-2021", "2021-2022", "2022-2023"]
    assert [len(window.records) for window in windows] == [4, 3, 2]
