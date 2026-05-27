"""Seldon Lab: temporal aggregation ontology and event workflows."""

# she-geofield is a required sibling package (not on PyPI).
# Install it with:  pip install -e ../she-geofield
# or for in-tree development set:  PYTHONPATH=src:../she-geofield/src
# The sys.path bootstrap that used to live here has been removed; explicit
# installation (or PYTHONPATH) is now the supported mechanism so that
# import resolution is predictable and reproducible.

__all__ = [
    "ontology",
    "datasets",
    "features",
    "events",
    "trajectories",
    "summaries",
    "laws",
    "experiments",
    "viz",
]
