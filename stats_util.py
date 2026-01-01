import math
from typing import Dict, List, Tuple
import numpy as np

def summary_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute descriptive statistics for a numeric array."""
    arr = np.asarray(values)
    n = arr.size
    if n == 0:
        return {k: math.nan for k in ["n", "mean", "sd", "skew", "exkurt", "min", "max"]}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=0))
    centered = arr - mean
    m2 = float(np.mean(centered ** 2))
    m3 = float(np.mean(centered ** 3))
    m4 = float(np.mean(centered ** 4))
    skew = math.nan
    exkurt = math.nan
    if m2 > 0:
        skew = m3 / (m2 ** 1.5)
        exkurt = m4 / (m2 * m2) - 3.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "skew": skew,
        "exkurt": exkurt,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

def autocorrelations(values: np.ndarray, k: int) -> np.ndarray:
    """Return autocorrelations for lags 1..k."""
    arr = np.asarray(values)
    n = arr.size
    if k <= 0 or n <= 1:
        return np.array([])
    k = min(k, n - 1)
    mean = arr.mean()
    centered = arr - mean
    denom = np.dot(centered, centered)
    if not (denom > 0):
        return np.full(k, math.nan)
    acf = []
    for lag in range(1, k + 1):
        num = np.dot(centered[lag:], centered[:-lag])
        acf.append(num / denom)
    return np.array(acf)

def print_summary_table(stats_dict: Dict[str, float], width: int = 15, precision: int = 4) -> None:
    """Pretty-print the summary statistics table."""
    headers = ["n", "mean", "sd", "skew", "ex_kurtosis", "min", "max"]
    print("         n" + "".join(f"{h:>{width}}" for h in headers[1:]))
    values = [stats_dict.get("n", math.nan), stats_dict.get("mean", math.nan), stats_dict.get("sd", math.nan),
              stats_dict.get("skew", math.nan), stats_dict.get("exkurt", math.nan), stats_dict.get("min", math.nan),
              stats_dict.get("max", math.nan)]
    line = f"{int(values[0]):10d}"
    for val in values[1:]:
        if math.isfinite(val):
            line += f"{val:{width}.{precision}f}"
        else:
            line += f"{'NA':>{width}}"
    print(line)

def print_autocorr_table(returns: np.ndarray, lags: int, width: int = 12, precision: int = 3) -> None:
    """Pretty-print the autocorrelation table for returns."""
    print(f"autocorrelations (lag 1-{lags})")
    headers = ["lag", "returns", "|returns|", "returns^2"]
    print(f"{headers[0]:>6} {headers[1]:>{width}} {headers[2]:>{width}} {headers[3]:>{width}}")
    abs_vals = np.abs(returns)
    sq_vals = returns ** 2
    ac_return = autocorrelations(returns, lags)
    ac_abs = autocorrelations(abs_vals, lags)
    ac_sq = autocorrelations(sq_vals, lags)
    for lag in range(1, lags + 1):
        def fmt(value: float) -> str:
            return f"{value:{width}.{precision}f}" if math.isfinite(value) else f"{'NA':>{width}}"
        print(f"{lag:6d} {fmt(ac_return[lag - 1] if ac_return.size >= lag else math.nan)}"
              f" {fmt(ac_abs[lag - 1] if ac_abs.size >= lag else math.nan)}"
              f" {fmt(ac_sq[lag - 1] if ac_sq.size >= lag else math.nan)}")

def compute_aicc(loglik: float, k: int, n: int) -> float:
    """Compute the finite-sample AICC criterion."""
    if n <= 0:
        return math.nan
    denom = n - k - 1
    if denom <= 0:
        return math.nan
    aic = 2.0 * k - 2.0 * loglik
    return aic + (2.0 * k * (k + 1)) / denom

def compute_bic(loglik: float, k: int, n: int) -> float:
    """Compute the Bayesian Information Criterion."""
    if n <= 0:
        return math.nan
    return math.log(n) * k - 2.0 * loglik

def sorted_indices(values: List[float], ascending: bool = True) -> List[int]:
    """Return indices that sort the values while handling NaNs."""
    finite = [math.isfinite(v) for v in values]
    order = list(range(len(values)))
    def key(idx: int):
        if not finite[idx]:
            return (1, idx)
        return (0, values[idx]) if ascending else (0, -values[idx])
    order.sort(key=key)
    return order

def compute_ranks(values: List[float], ascending: bool = True) -> List[int]:
    """Convert metric values to ordinal ranks."""
    order = sorted_indices(values, ascending)
    ranks = [len(values)] * len(values)
    for pos, idx in enumerate(order):
        ranks[idx] = pos + 1
    return ranks

def print_selects(label: str, names: List[str], values: List[float]) -> None:
    """Print a ranking summary for a single metric."""
    order = sorted_indices(values, ascending=True)
    print(f"\n{label} selects")
    if not order:
        print("  (no models)")
        return
    print(f"{'model':<25} {label:>18} {'diff':>18}")
    base = values[order[0]]
    for pos, idx in enumerate(order):
        name = names[idx]
        val = values[idx]
        diff = val - base if math.isfinite(val) and math.isfinite(base) else math.nan
        val_str = f"{val:18.6f}" if math.isfinite(val) else f"{'NA':>18}"
        diff_str = f"{diff:18.6f}" if math.isfinite(diff) else f"{'NA':>18}"
        suffix = "  best" if pos == 0 else ""
        print(f"{name:<25} {val_str} {diff_str}{suffix}")

def residual_summary_table(entries: List[Tuple[str, np.ndarray]], width: int = 15, precision: int = 4) -> None:
    """Print summary stats for standardized residuals."""
    print("\nstandardized residual stats")
    header = ["model", "n", "mean", "sd", "skew", "ex_kurtosis", "min", "max"]
    print(f"{header[0]:<25} {header[1]:>10}" + "".join(f"{h:>{width}}" for h in header[2:]))
    for name, resid in entries:
        stats = summary_stats(resid)
        line = f"{name:<25} {int(stats['n']):>10d}"
        for key in ["mean", "sd", "skew", "exkurt", "min", "max"]:
            val = stats[key]
            if math.isfinite(val):
                line += f"{val:{width}.{precision}f}"
            else:
                line += f"{'NA':>{width}}"
        print(line)

def cond_sd_summary_table(entries: List[Tuple[str, np.ndarray]], width: int = 15, precision: int = 4) -> None:
    """Print summary stats for conditional standard deviations."""
    print("\nconditional sd stats")
    header = ["model", "n", "mean", "sd", "skew", "ex_kurtosis", "min", "max"]
    print(f"{header[0]:<25} {header[1]:>10}" + "".join(f"{h:>{width}}" for h in header[2:]))
    for name, cond_sd in entries:
        stats = summary_stats(cond_sd)
        line = f"{name:<25} {int(stats['n']):>10d}"
        for key in ["mean", "sd", "skew", "exkurt", "min", "max"]:
            val = stats[key]
            if math.isfinite(val):
                line += f"{val:{width}.{precision}f}"
            else:
                line += f"{'NA':>{width}}"
        print(line)
