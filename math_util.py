import math

HUGE_PENALTY = 1e12

def sigmoid(x: float) -> float:
    """Map a real-valued input to (0, 1) via the logistic transform."""
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p: float) -> float:
    """Map a probability in (0, 1) to the real line."""
    if p <= 0:
        return -HUGE_PENALTY
    if p >= 1:
        return HUGE_PENALTY
    return math.log(p / (1.0 - p))