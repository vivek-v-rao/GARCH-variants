import numpy as np
from scipy import optimize

maxiter_max = 1000000
fatol = 1e-8 # Function Value Absolute Tolerance

def minimize_with_simplex(func, theta0: np.ndarray, steps: np.ndarray, maxiter: int,
    verbose=False) -> np.ndarray:
    """Run a Nelder-Mead search with explicit step sizes."""
    steps = np.where(steps > 0, steps, 0.1)
    simplex = [theta0]
    for i in range(theta0.size):
        vertex = theta0.copy()
        vertex[i] += steps[i]
        simplex.append(vertex)
    result = optimize.minimize(
        func,
        theta0,
        method="Nelder-Mead",
        options={"maxiter": min(maxiter, maxiter_max), "fatol": fatol, "xatol": 1e-8, "initial_simplex": np.array(simplex)},
    )
    if verbose:
        print(f"Nelder-Mead iterations: {result.nit}, function evals: {result.nfev}")
    return result.x if result.x.size else theta0