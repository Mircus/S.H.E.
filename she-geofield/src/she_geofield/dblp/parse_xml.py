"""Streaming DBLP XML parsing for papers only."""

from collections.abc import Iterator
import csv
import gzip
from pathlib import Path

from lxml import etree as ET

from .authors import canonicalize_authors
from .records import PaperRecord


SUPPORTED_RECORD_TYPES = {"article", "inproceedings"}


def _open_xml_stream(xml_path: str | Path):
    path = Path(xml_path)
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _extract_text(elem: ET.Element, tag: str) -> str | None:
    child = elem.find(tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None


def iter_dblp_records(
    xml_path: str | Path,
    publication_types: set[str] | None = None,
) -> Iterator[PaperRecord]:
    supported_types = publication_types or SUPPORTED_RECORD_TYPES
    with _open_xml_stream(xml_path) as handle:
        context = ET.iterparse(
            handle,
            events=("end",),
            recover=True,
            resolve_entities=False,
            load_dtd=True,
            no_network=False,
            huge_tree=True,
        )
        for _event, elem in context:
            if elem.tag not in supported_types:
                continue

            year_text = _extract_text(elem, "year")
            if year_text is None or not year_text.isdigit():
                elem.clear()
                continue

            authors = canonicalize_authors(
                [child.text or "" for child in elem.findall("author")]
            )
            if not authors:
                elem.clear()
                continue

            yield PaperRecord(
                key=elem.attrib.get("key", ""),
                year=int(year_text),
                authors=authors,
                venue=_extract_text(elem, "journal") or _extract_text(elem, "booktitle"),
                pub_type=elem.tag,
                title=_extract_text(elem, "title"),
            )
            elem.clear()


def load_dblp_records(
    xml_path: str | Path,
    publication_types: set[str] | None = None,
) -> list[PaperRecord]:
    return list(iter_dblp_records(xml_path, publication_types=publication_types))


def load_csv_records(csv_path: str | Path) -> list[PaperRecord]:
    records: list[PaperRecord] = []
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            authors = canonicalize_authors((row.get("authors") or "").split("|"))
            if not authors:
                continue
            records.append(
                PaperRecord(
                    key=row.get("key", ""),
                    year=int(row["year"]),
                    authors=authors,
                    venue=row.get("venue") or None,
                    pub_type=row.get("pub_type") or None,
                    title=row.get("title") or None,
                )
            )
    return records
