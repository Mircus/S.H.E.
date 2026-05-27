from dataclasses import dataclass

# Law-candidate model inventory note:
# "author_level_proxy" is defined in she_geofield.dblp.temporal_flow and scores
# each aggregation by its internal_activity (sum of co-authorship weights among
# member pairs inside the unit).  It acts as a node/edge-level baseline: it
# measures raw co-authorship concentration without geometric structure.
# See also: aggregation_state_vector() in she_geofield.dblp.metrics for the
# definition of the underlying "activity" state field.


@dataclass(frozen=True)
class LawCandidate:
    name: str
    description: str
    confidence: str
    evidence: str = ""
    counterexamples: str = ""


def birth_law_candidate_from_rows(rows: list[dict[str, str]]) -> LawCandidate:
    birth_rows = [row for row in rows if row["event_type"] == "birth"]
    venues = ", ".join(sorted(row["venue"] for row in birth_rows))
    evidence = "; ".join(
        f"{row['venue']}: {row['best_predictor']} (rho={float(row['best_predictor_rho']):.4f}, events={row['event_count']})"
        for row in birth_rows
    )
    return LawCandidate(
        name="aggregation_birth",
        description=(
            "Aggregation birth occurs when local closure and sustained activity "
            "convert a weak persistence seed into a stable higher-order unit."
        ),
        confidence="moderate",
        evidence=f"Cross-venue support on {venues}: {evidence}",
        counterexamples=(
            "Thin edge seeds with zero closure and weak activity usually fail "
            "to become bona fide aggregations."
        ),
    )
