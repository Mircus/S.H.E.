"""Bridge-to-cluster event type constant and stub.

Architecture note (Critic Issue R2 / Issue 12):
  Detection of bridge-to-cluster candidates is implemented in the
  she-geofield layer:
    - ``she_geofield.dblp.metrics.bridge_to_cluster_score``
    - ``she_geofield.dblp.metrics.bridge_to_cluster_candidates``
    - configs: ``she-geofield/configs/dblp_sdm_bridge_to_cluster.yaml``

  This seldon-lab module owns the *event classification* layer: deciding
  which detected candidates satisfy the ontological criteria for a
  bona fide bridge-to-cluster event.  That classification logic is
  planned but not yet implemented.  The stub is intentional: it marks the
  boundary between she-geofield (detection) and seldon-lab (classification)
  in the architecture without pretending the classification is done.

Status: PLANNED — detection in she-geofield, classification logic here TBD.
"""

EVENT_TYPE = "bridge_to_cluster"
