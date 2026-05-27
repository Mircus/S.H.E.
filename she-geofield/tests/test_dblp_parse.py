from pathlib import Path

from she_geofield.dblp.parse_xml import load_dblp_records


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dblp_sample.xml"


def test_parse_dblp_sample_records():
    records = load_dblp_records(FIXTURE)

    assert len(records) == 6
    assert records[0].key == "conf/test/alpha2020"
    assert records[1].authors == ["Alice", "Bob", "Carol"]
    assert {record.pub_type for record in records} == {"article", "inproceedings"}
