from pathlib import Path

from she_geofield.dblp.lift import build_window_complex
from she_geofield.dblp.parse_xml import load_dblp_records
from she_geofield.dblp.records import TimeWindow


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dblp_sample.xml"


def test_contained_vs_exact_team_weights():
    records = load_dblp_records(FIXTURE)
    window = TimeWindow(2020, 2020, tuple(record for record in records if record.year == 2020))

    contained = build_window_complex(window, weight_mode="contained")
    exact = build_window_complex(window, weight_mode="exact")

    assert contained.contained_support[("Alice", "Bob")] == 2.0
    assert exact.exact_support[("Alice", "Bob", "Carol")] == 1.0
    assert exact.exact_support.get(("Alice", "Bob"), 0.0) == 1.0
    assert contained.triangle_weights[("Alice", "Bob", "Carol")] == 1.0
