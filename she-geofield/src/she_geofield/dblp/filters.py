import re

from .records import PaperRecord


def filter_records(
    records: list[PaperRecord],
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    min_team_size: int | None = None,
    max_team_size: int | None = None,
    venue_list: list[str] | None = None,
    venue_regex: str | None = None,
    publication_types: list[str] | None = None,
) -> list[PaperRecord]:
    allowed_venues = {venue.lower() for venue in venue_list or []}
    venue_pattern = re.compile(venue_regex, flags=re.IGNORECASE) if venue_regex else None
    allowed_types = {pub_type.lower() for pub_type in publication_types or []}

    filtered: list[PaperRecord] = []
    for record in records:
        team_size = len(record.authors)
        if start_year is not None and record.year < start_year:
            continue
        if end_year is not None and record.year > end_year:
            continue
        if min_team_size is not None and team_size < min_team_size:
            continue
        if max_team_size is not None and team_size > max_team_size:
            continue
        if allowed_types and (record.pub_type or "").lower() not in allowed_types:
            continue

        venue = (record.venue or "").lower()
        if allowed_venues and venue not in allowed_venues:
            continue
        if venue_pattern and not venue_pattern.search(record.venue or ""):
            continue
        filtered.append(record)
    return filtered
