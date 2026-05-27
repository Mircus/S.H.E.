from pathlib import Path


def write_law_candidate_report(
    path: str | Path,
    *,
    title: str,
    candidate_regularity: str,
    cross_venue_evidence: str,
    counterexamples: str,
    confidence: str,
    unclear: str,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                "## Candidate regularity",
                candidate_regularity,
                "",
                "## Cross-venue evidence",
                cross_venue_evidence,
                "",
                "## Counterexamples",
                counterexamples,
                "",
                "## Confidence level",
                confidence,
                "",
                "## What remains unclear",
                unclear,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output
