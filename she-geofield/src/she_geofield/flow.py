from copy import deepcopy
import numpy as np
from scipy.linalg import expm

from .hodge import hodge_laplacian_1, hodge_laplacian_2
from .curvature import triangle_curvature, edge_forman_curvature
from .toy_complex import WeightedToyComplex

def fixed_geometry_step(
    complex_: WeightedToyComplex,
    x1: np.ndarray,
    dt: float = 0.5,
    location: str = "edges",
) -> np.ndarray:
    if location == "edges":
        laplacian = hodge_laplacian_1(complex_)
    elif location == "triangles":
        laplacian = hodge_laplacian_2(complex_)
    else:
        raise ValueError(f"unsupported field location: {location}")
    P = expm(-dt * laplacian)
    return P @ x1

def geometry_step(complex_: WeightedToyComplex, x1: np.ndarray = None,
                  eta: float = 0.1, lam: float = 0.5, mu: float = 0.0,
                  field_location: str = "edges") -> None:
    """Evolve triangle weights by field-coupled curvature-driven flow.

    Uses exponential update w_t <- w_t exp(-eta G(t)) to guarantee positivity.
    When mu > 0, the field state x1 feeds back into the curvature: triangles
    carrying more signal have lower effective curvature (weight grows).
    Edge weights are held fixed (edge Forman curvature is diagnostic only).
    """
    min_log_factor = np.log(np.finfo(float).tiny)
    max_log_factor = np.log(np.finfo(float).max)
    for tri in complex_.triangles:
        F = triangle_curvature(
            complex_,
            tri,
            x1=x1,
            lam=lam,
            mu=mu,
            field_location=field_location,
        )
        # Clamp only for numerical stability; the analytic update stays positive.
        log_factor = np.clip(-eta * F, min_log_factor, max_log_factor)
        complex_.triangle_weights[tri] *= np.exp(log_factor)

def coupled_step(complex_: WeightedToyComplex, x1: np.ndarray,
                 dt: float = 0.5, eta: float = 0.1, lam: float = 0.5,
                 mu: float = 0.0, field_location: str = "edges"):
    x_next = fixed_geometry_step(complex_, x1, dt=dt, location=field_location)
    geometry_step(complex_, x1=x1, eta=eta, lam=lam, mu=mu, field_location=field_location)
    return x_next

def iterate_coupled(complex_: WeightedToyComplex, x1: np.ndarray,
                    steps: int = 8, dt: float = 0.5, eta: float = 0.05,
                    lam: float = 0.5, mu: float = 0.0,
                    field_location: str = "edges"):
    traj = [x1.copy()]
    tri_weights = [deepcopy(complex_.triangle_weights)]
    edge_weights = [deepcopy(complex_.edge_weights)]
    cur = x1.copy()
    for _ in range(steps):
        cur = coupled_step(
            complex_,
            cur,
            dt=dt,
            eta=eta,
            lam=lam,
            mu=mu,
            field_location=field_location,
        )
        traj.append(cur.copy())
        tri_weights.append(deepcopy(complex_.triangle_weights))
        edge_weights.append(deepcopy(complex_.edge_weights))
    return traj, tri_weights, edge_weights
