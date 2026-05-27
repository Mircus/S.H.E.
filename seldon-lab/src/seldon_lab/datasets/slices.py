from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSlice:
    name: str
    start_year: int
    end_year: int
    window_width: int
    window_stride: int
