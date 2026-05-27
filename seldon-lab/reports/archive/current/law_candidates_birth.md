# Law Candidates: Birth

## Candidate regularity

An aggregation birth event is most likely when an already closed local
higher-order configuration carries enough current activity to convert a weak
persistence seed into stable persistence across the next window.

## Cross-venue evidence

Current cross-venue evidence comes from the DBLP `SDM 2018--2024` and
`WSDM 2018--2024` slices, using rolling 3-year windows with stride 1.

- `SDM`: `467` birth events, best predictor `aggregation_birth_score`, mean
  Spearman `0.2000`
- `WSDM`: `952` birth events, best predictor `aggregation_birth_score`, mean
  Spearman `0.2368`

The positive birth cases are mostly triangle seeds that begin with closure
already present and then move from persistence `0.5` to `1.0` while activity
rises in the next window.

## Counterexamples

Thin edge seeds with closure `0.0` and weak or collapsing activity usually do
not become bona fide aggregations. These failed cases show that activity alone
is not enough without local closure.

## Confidence level

Moderate. The signal is stable across two venues, but the ontology is still
young and the current implementation is more reliable for simplex seeds than
for richer local neighborhood aggregations.

## What remains unclear

- whether neighborhood aggregations sharpen the birth signature further
- whether the same regularity survives outside data-mining venues
- whether boundary role should enter the birth rule as a genuine threshold or
  only as a secondary modifier
