"""Fission event type constant and stub.

Architecture note (Critic Issue R2 / Issue 12):
  Fission — the splitting of a single aggregation into two or more
  descendant aggregations — is a declared event type in the seldon-lab
  ontology.  No detection mechanism exists at any level (neither in
  she-geofield nor here).  This stub marks the intended boundary for
  future implementation.

Status: PLANNED — no detection logic exists anywhere in the codebase yet.
"""

EVENT_TYPE = "fission"
