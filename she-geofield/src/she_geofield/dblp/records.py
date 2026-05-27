from dataclasses import dataclass, field
from typing import Any


Simplex = tuple[str, ...]
Edge = tuple[str, str]
Triangle = tuple[str, str, str]
AggregationId = str


@dataclass(frozen=True)
class PaperRecord:
    key: str
    year: int
    authors: list[str]
    venue: str | None
    pub_type: str | None
    title: str | None


@dataclass(frozen=True)
class TimeWindow:
    start_year: int
    end_year: int
    records: tuple[PaperRecord, ...]

    @property
    def label(self) -> str:
        return f"{self.start_year}-{self.end_year}"


@dataclass
class WindowSummary:
    window_label: str
    paper_count: int
    author_count: int
    simplex_count_by_dim: dict[int, int]
    density_stats: dict[str, float]


@dataclass
class DblpSimplicialComplex:
    vertices: list[str]
    edges: list[Edge]
    triangles: list[Triangle]
    vertex_weights: dict[str, float]
    edge_weights: dict[Edge, float]
    triangle_weights: dict[Triangle, float]
    exact_support: dict[Simplex, float] = field(default_factory=dict)
    contained_support: dict[Simplex, float] = field(default_factory=dict)
    supporting_papers: dict[Simplex, tuple[str, ...]] = field(default_factory=dict)
    simplex_data: dict[Simplex, dict[str, Any]] = field(default_factory=dict)
    triangle_cohesion: dict[Triangle, float] = field(default_factory=dict)
    window_start: int | None = None
    window_end: int | None = None
    records: tuple[PaperRecord, ...] = field(default_factory=tuple)
    weight_mode: str = "contained"

    @property
    def label(self) -> str:
        if self.window_start is None or self.window_end is None:
            return "unknown"
        return f"{self.window_start}-{self.window_end}"

    def support_map(self, mode: str | None = None) -> dict[Simplex, float]:
        chosen = mode or self.weight_mode
        if chosen == "exact":
            return self.exact_support
        return self.contained_support


@dataclass(frozen=True)
class AggregationSnapshot:
    aggregation_id: AggregationId
    unit_type: str
    members: tuple[str, ...]
    anchor_simplex: tuple[str, ...]
    window_label: str
    state: dict[str, float]
