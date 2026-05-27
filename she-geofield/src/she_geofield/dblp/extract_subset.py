import argparse
import csv
from pathlib import Path

from .filters import filter_records
from .parse_xml import iter_dblp_records


def extract_subset(
    input_file: str | Path,
    output_file: str | Path,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    min_team_size: int | None = None,
    max_team_size: int | None = None,
    venue_regex: str | None = None,
    publication_types: list[str] | None = None,
    max_records: int | None = None,
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, object]] = []
    for record in iter_dblp_records(
        input_file,
        publication_types=set(publication_types) if publication_types else None,
    ):
        filtered = filter_records(
            [record],
            start_year=start_year,
            end_year=end_year,
            min_team_size=min_team_size,
            max_team_size=max_team_size,
            venue_regex=venue_regex,
            publication_types=publication_types,
        )
        if not filtered:
            continue
        selected.append(
            {
                "key": record.key,
                "year": record.year,
                "venue": record.venue or "",
                "pub_type": record.pub_type or "",
                "title": record.title or "",
                "authors": "|".join(record.authors),
            }
        )
        if max_records is not None and len(selected) >= max_records:
            break

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["key", "year", "venue", "pub_type", "title", "authors"],
        )
        writer.writeheader()
        writer.writerows(selected)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a filtered DBLP subset to CSV.")
    parser.add_argument("--input", required=True, help="Path to dblp.xml or dblp.xml.gz")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--min-team-size", type=int)
    parser.add_argument("--max-team-size", type=int)
    parser.add_argument("--venue-regex")
    parser.add_argument("--publication-types", default="article,inproceedings")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    output = extract_subset(
        args.input,
        args.output,
        start_year=args.start_year,
        end_year=args.end_year,
        min_team_size=args.min_team_size,
        max_team_size=args.max_team_size,
        venue_regex=args.venue_regex,
        publication_types=[item for item in args.publication_types.split(",") if item],
        max_records=args.max_records,
    )
    print(f"Wrote filtered subset to {output}")


if __name__ == "__main__":
    main()
