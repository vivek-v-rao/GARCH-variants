import math
K_PI = math.pi

def theta_to_dof(theta_val: float) -> float:
    """Convert an unconstrained parameter into a valid Student t degrees-of-freedom value."""
    return 2.0 + math.exp(theta_val)

def log_t_pdf(x: float, dof: float) -> float:
    """Compute the log-density of a Student t random variable."""
    a = math.lgamma(0.5 * (dof + 1.0)) - math.lgamma(0.5 * dof)
    b = -0.5 * (math.log(dof) + math.log(K_PI))
    c = -0.5 * (dof + 1.0) * math.log(1.0 + (x * x) / dof)
    return a + b + c

def standardized_t_scale(dof: float) -> float:
    """Return the scaling constant for standardized t shocks."""
    return math.sqrt(dof / (dof - 2.0))