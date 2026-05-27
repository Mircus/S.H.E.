from .candidates import LawCandidate


def summarize_candidate(candidate: LawCandidate) -> str:
    return "\n".join(
        [
            f"# {candidate.name}",
            "",
            "## Candidate regularity",
            candidate.description,
            "",
            "## Evidence",
            candidate.evidence,
            "",
            "## Counterexamples",
            candidate.counterexamples,
            "",
            "## Confidence",
            candidate.confidence,
            "",
        ]
    )
