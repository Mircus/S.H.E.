import numpy as np
from .toy_complex import WeightedToyComplex

def right_region_edges(complex_: WeightedToyComplex):
    return [i for i, e in enumerate(complex_.edges) if "e" in e or "d" in e or "f" in e]

def right_region_mass(complex_: WeightedToyComplex, x1: np.ndarray) -> float:
    idx = right_region_edges(complex_)
    return float(np.sum(x1[idx]))
