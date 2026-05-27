from pathlib import Path

import numpy as np

from she_geofield.dblp.fields import collaboration_activation_field
from she_geofield.dblp.lift import build_window_complex
from she_geofield.dblp.parse_xml import load_dblp_records
from she_geofield.dblp.records import TimeWindow


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dblp_sample.xml"


def test_collaboration_activation_field_tracks_edge_support():
    records = load_dblp_records(FIXTURE)
    window = TimeWindow(2020, 2021, tuple(record for record in records if record.year <= 2021))
    complex_ = build_window_complex(window, weight_mode="contained")

    x = collaboration_activation_field(complex_, location="edges", mode="support_count")

    assert x.shape == (len(complex_.edges),)
    assert np.all(x >= 0.0)
    edge_to_idx = {edge: idx for idx, edge in enumerate(complex_.edges)}
    assert x[edge_to_idx[("Alice", "Bob")]] > 0.0


def test_collaboration_activation_field_supports_triangles():
    records = load_dblp_records(FIXTURE)
    window = TimeWindow(2020, 2021, tuple(record for record in records if record.year <= 2021))
    complex_ = build_window_complex(window, weight_mode="contained")

    x = collaboration_activation_field(complex_, location="triangles", mode="support_count")

    assert x.shape == (len(complex_.triangles),)
    assert np.all(x >= 0.0)
