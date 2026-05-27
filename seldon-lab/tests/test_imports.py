from seldon_lab.ontology.state import AggregationState, is_bona_fide_aggregation


def test_bona_fide_aggregation_rule():
    state = AggregationState(
        closure=0.5,
        persistence=0.5,
        activity=1.5,
        boundary_role=0.4,
        growth_potential=1.2,
        support=1.0,
    )
    assert is_bona_fide_aggregation(state)
