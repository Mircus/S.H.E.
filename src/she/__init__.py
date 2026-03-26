"""SHE -- Simplicial Hyperstructure Engine.

A source-available research library for modeling and analyzing decorated
higher-order relational structures, with a current computational focus on
weighted simplicial representations and social/group-level diffusion analysis.

Public API
----------
>>> from she import SHEHyperstructure, rank_diffusers, find_bridge_simplices
>>> from she import SHEConfig, SHESimplicialComplex, SHEDataLoader
>>> from she import SHEHodgeDiffusion, DiffusionResult
"""

# -- core layer -----------------------------------------------------------
from .config import SHEConfig
from .types import DiffusionResult
from .complex import SHESimplicialComplex
from .io import SHEDataLoader
from .diffusion import SHEHodgeDiffusion
from .visualize import SHEDiffusionVisualizer
from .engine import SHEEngine

# -- modeling layer -------------------------------------------------------
from .hyperstructure import SHEHyperstructure

# -- social analysis layer ------------------------------------------------
from .social import (
    rank_diffusers,
    rank_entity_diffusers,
    rank_simplex_diffusers,
    find_bridge_simplices,
    group_cohesion,
    rank_influencers,
    RankedItem,
    BridgeSimplex,
    CohesionScore,
)

# -- temporal layer -------------------------------------------------------
from .temporal import window, rolling_windows, decay_window

# -- export layer ---------------------------------------------------------
from .export import (
    ranked_items_to_csv,
    ranked_items_to_json,
    bridges_to_csv,
    bridges_to_json,
    cohesion_to_csv,
    cohesion_to_json,
)

__version__ = "0.1.2"

__all__ = [
    # core
    "SHEConfig",
    "DiffusionResult",
    "SHESimplicialComplex",
    "SHEDataLoader",
    "SHEHodgeDiffusion",
    "SHEDiffusionVisualizer",
    "SHEEngine",
    # modeling
    "SHEHyperstructure",
    # social analysis
    "rank_diffusers",
    "rank_entity_diffusers",
    "rank_simplex_diffusers",
    "find_bridge_simplices",
    "group_cohesion",
    "rank_influencers",
    "RankedItem",
    "BridgeSimplex",
    "CohesionScore",
    # temporal
    "window",
    "rolling_windows",
    "decay_window",
    # export
    "ranked_items_to_csv",
    "ranked_items_to_json",
    "bridges_to_csv",
    "bridges_to_json",
    "cohesion_to_csv",
    "cohesion_to_json",
]
