from typing import Dict, List, Sequence, Optional
from dataclasses import dataclass
import math
import numpy as np
from arch.univariate import (ConstantMean, EGARCH, GARCH,
    StudentsT, Normal, SkewStudent)
from math_util import sigmoid, logit, HUGE_PENALTY
from stats_util import sorted_indices
from dist import log_t_pdf, standardized_t_scale
from minimize import minimize_with_simplex, maxiter_max
from scipy import optimize
from dist import theta_to_dof

K_PI = math.pi

def _run_optimizer(func, theta0, steps, maxiter, optimizer):
    method = (optimizer or 'nelder-mead').lower()
    if method == 'nelder-mead':
        return minimize_with_simplex(func, theta0, steps, maxiter)
    if method == 'bfgs':
        options = {"maxiter": min(maxiter, maxiter_max)}
        result = optimize.minimize(func, theta0, method='L-BFGS-B', options=options)
        return result.x if result.x.size else theta0
    raise ValueError(f"unknown optimizer: {optimizer}")


@dataclass
class ModelRow: # class to store GARCH parameters and outputs
    label: str
    mu: float = math.nan
    omega: float = math.nan
    alpha: float = math.nan
    beta: float = math.nan
    gamma: float = math.nan
    shift: float = math.nan
    skew: float = math.nan
    dof: float = math.nan
    uncond_sd: float = math.nan
    loglik: float = math.nan
    n_params: int = 0
    cond_sd: Optional[np.ndarray] = None
    std_resid: Optional[np.ndarray] = None

def fit_nagarch(r: np.ndarray, dist: str, dof0: float = 6.0, warm: Optional[np.ndarray] = None, optimizer: str = 'nelder-mead'):
    """Fit a NAGARCH(1,1) model under the requested distribution."""
    if r.size <= 5:
        raise ValueError("fit_nagarch requires more data")
    mean = float(r.mean())
    var = float(np.mean((r - mean) ** 2))
    if not (var > 0.0):
        var = 1e-8
    u0 = 0.97
    v0 = 0.50
    g0 = 0.0
    omega0 = var * (1.0 - u0)
    if not (omega0 > 0.0):
        omega0 = 1e-10
    sd = math.sqrt(var)
    if dist == "normal":
        steps = np.array([0.10 * sd, 0.50, 0.50, 0.50, 0.20])
        theta0 = np.array([mean, math.log(omega0), logit(u0), logit(v0), g0])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: nagarch_neg_loglik(r, theta_to_nagarch_params(th), dist, 0.0)
        th_star = _run_optimizer(func, theta0, steps, 3000, optimizer)
        params = theta_to_nagarch_params(th_star)
        params["skew"] = 0.0
        return params, 0.0, func(th_star)
    if dist == "student-t":
        if not (dof0 > 2.0001):
            dof0 = 6.0
        steps = np.array([0.10 * sd, 0.50, 0.50, 0.50, 0.20, 0.30])
        theta0 = np.array([mean, math.log(omega0), logit(u0), logit(v0), g0, math.log(dof0 - 2.0)])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: nagarch_neg_loglik(r, theta_to_nagarch_params(th[:-1]), dist, theta_to_dof(th[-1]))
        th_star = _run_optimizer(func, theta0, steps, 4000, optimizer)
        params = theta_to_nagarch_params(th_star[:-1])
        params["skew"] = 0.0
        dof = theta_to_dof(th_star[-1])
        return params, dof, func(th_star)
    if dist == "skew-student":
        if not (dof0 > 2.0001):
            dof0 = 6.0
        steps = np.array([0.10 * sd, 0.50, 0.50, 0.50, 0.20, 0.30, 0.30])
        theta0 = np.array([mean, math.log(omega0), logit(u0), logit(v0), g0, math.log(dof0 - 2.0), 0.0])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: nagarch_neg_loglik(r, theta_to_nagarch_params(th[:-2]), dist, theta_to_dof(th[-2]), theta_to_skew(th[-1]))
        th_star = _run_optimizer(func, theta0, steps, 4500, optimizer)
        params = theta_to_nagarch_params(th_star[:-2])
        params["skew"] = theta_to_skew(th_star[-1])
        dof = theta_to_dof(th_star[-2])
        return params, dof, func(th_star)
    raise ValueError("invalid dist")


def fit_igarch(r: np.ndarray, dist: str, dof0: float = 6.0, warm: Optional[np.ndarray] = None, optimizer: str = 'nelder-mead'):
    """Fit an IGARCH(1,1) model under the requested distribution."""
    if r.size <= 5:
        raise ValueError("fit_igarch requires more data")
    mean = float(r.mean())
    var = float(np.mean((r - mean) ** 2))
    if not (var > 0.0):
        var = 1e-8
    sd = math.sqrt(var)
    alpha0 = 0.1
    if dist == "normal":
        steps = np.array([0.10 * sd, 0.50])
        theta0 = np.array([mean, logit(alpha0)])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: igarch_neg_loglik(r, theta_to_igarch_params(th), dist, 0.0)
        th_star = _run_optimizer(func, theta0, steps, 2000, optimizer)
        params = theta_to_igarch_params(th_star)
        params["skew"] = 0.0
        return params, 0.0, func(th_star)
    if dist == "student-t":
        if not (dof0 > 2.0001):
            dof0 = 6.0
        steps = np.array([0.10 * sd, 0.50, 0.30])
        theta0 = np.array([mean, logit(alpha0), math.log(dof0 - 2.0)])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: igarch_neg_loglik(
            r, theta_to_igarch_params(th[:-1]), dist, theta_to_dof(th[-1])
        )
        th_star = _run_optimizer(func, theta0, steps, 2500, optimizer)
        params = theta_to_igarch_params(th_star[:-1])
        params["skew"] = 0.0
        dof = theta_to_dof(th_star[-1])
        return params, dof, func(th_star)
    if dist == "skew-student":
        if not (dof0 > 2.0001):
            dof0 = 6.0
        steps = np.array([0.10 * sd, 0.50, 0.30, 0.30])
        theta0 = np.array([mean, logit(alpha0), math.log(dof0 - 2.0), 0.0])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: igarch_neg_loglik(
            r, theta_to_igarch_params(th[:-2]), dist, theta_to_dof(th[-2]), theta_to_skew(th[-1])
        )
        th_star = _run_optimizer(func, theta0, steps, 3000, optimizer)
        params = theta_to_igarch_params(th_star[:-2])
        params["skew"] = theta_to_skew(th_star[-1])
        dof = theta_to_dof(th_star[-2])
        return params, dof, func(th_star)
    raise ValueError("invalid dist")


def fit_st(r: np.ndarray, dist: str, dof0: float = 6.0, warm: Optional[np.ndarray] = None, optimizer: str = 'nelder-mead'):
    """Fit an ST-GARCH model under the requested distribution."""
    if r.size <= 5:
        raise ValueError("fit_st requires more data")
    mean = float(r.mean())
    var = float(np.mean((r - mean) ** 2))
    if not (var > 0.0):
        var = 1e-8
    u0 = 0.97
    v0 = 0.50
    w0 = 0.50
    shift0 = 0.0
    omega0 = var * (1.0 - u0)
    if not (omega0 > 0.0):
        omega0 = 1e-10
    sd = math.sqrt(var)
    if dist == "normal":
        steps = np.array([0.10 * sd, 0.50, 0.50, 0.50, 0.50, 0.10])
        theta0 = np.array([mean, math.log(omega0), logit(u0), logit(v0), logit(w0), shift0])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: st_neg_loglik(r, theta_to_st_params(th), dist, 0.0)
        th_star = _run_optimizer(func, theta0, steps, 4000, optimizer)
        params = theta_to_st_params(th_star)
        params["skew"] = 0.0
        return params, 0.0, func(th_star)
    if dist == "student-t":
        if not (dof0 > 2.0001):
            dof0 = 6.0
        steps = np.array([0.10 * sd, 0.50, 0.50, 0.50, 0.50, 0.10, 0.30])
        theta0 = np.array([mean, math.log(omega0), logit(u0), logit(v0), logit(w0), shift0, math.log(dof0 - 2.0)])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: st_neg_loglik(r, theta_to_st_params(th[:-1]), dist, theta_to_dof(th[-1]))
        th_star = _run_optimizer(func, theta0, steps, 4500, optimizer)
        params = theta_to_st_params(th_star[:-1])
        params["skew"] = 0.0
        dof = theta_to_dof(th_star[-1])
        return params, dof, func(th_star)
    if dist == "skew-student":
        if not (dof0 > 2.0001):
            dof0 = 6.0
        steps = np.array([0.10 * sd, 0.50, 0.50, 0.50, 0.50, 0.10, 0.30, 0.30])
        theta0 = np.array([mean, math.log(omega0), logit(u0), logit(v0), logit(w0), shift0, math.log(dof0 - 2.0), 0.0])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: st_neg_loglik(r, theta_to_st_params(th[:-2]), dist, theta_to_dof(th[-2]), theta_to_skew(th[-1]))
        th_star = _run_optimizer(func, theta0, steps, 5000, optimizer)
        params = theta_to_st_params(th_star[:-2])
        params["skew"] = theta_to_skew(th_star[-1])
        dof = theta_to_dof(th_star[-2])
        return params, dof, func(th_star)
    raise ValueError("invalid dist")

def theta_to_nagarch_params(theta: Sequence[float]) -> Dict[str, float]:
    """Convert unconstrained theta values into constrained NAGARCH parameters."""
    mu = theta[0]
    omega = math.exp(theta[1])
    u = sigmoid(theta[2])
    v = sigmoid(theta[3])
    gamma = theta[4]
    beta = u * v
    alpha = u * (1.0 - v) / (1.0 + gamma * gamma)
    return {"mu": mu, "omega": omega, "alpha": alpha, "beta": beta, "gamma": gamma}



def theta_to_igarch_params(theta: Sequence[float]) -> Dict[str, float]:
    """Convert unconstrained theta values into constrained IGARCH parameters."""
    mu = theta[0]
    alpha_raw = sigmoid(theta[1])
    eps = 1e-6
    alpha = eps + (1.0 - 2.0 * eps) * alpha_raw
    beta = 1.0 - alpha
    return {"mu": mu, "alpha": alpha, "beta": beta}


def theta_to_st_params(theta: Sequence[float]) -> Dict[str, float]:
    """Convert unconstrained theta values into constrained ST-GARCH parameters."""
    mu = theta[0]
    omega = math.exp(theta[1])
    u = sigmoid(theta[2])
    v = sigmoid(theta[3])
    w = sigmoid(theta[4])
    remainder = u * (1.0 - v)
    beta = u * v
    gamma = 2.0 * remainder * w
    alpha = remainder * (1.0 - w)
    shift = theta[5]
    return {
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "shift": shift,
    }

def theta_to_skew(theta_val: float) -> float:
    """Map an unconstrained value into (-0.99, 0.99) for skew parameters."""
    return math.tanh(theta_val) * 0.99

def initial_variance(r: np.ndarray, mu: float) -> float:
    """Compute a fallback variance estimate from centered data."""
    var = float(np.mean((r - mu) ** 2))
    if not (var > 1e-12):
        var = 1e-12
    return var


def nagarch_neg_loglik(r: np.ndarray, params: Dict[str, float], dist: str, dof: float, skew: float = 0.0) -> float:
    """Evaluate the negative log-likelihood for the NAGARCH model."""
    if r.size == 0:
        return HUGE_PENALTY
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    mu = params["mu"]
    if not (omega > 0.0 and alpha >= 0.0 and beta >= 0.0):
        return HUGE_PENALTY
    u = alpha * (1.0 + gamma * gamma) + beta
    if not (u < 1.0):
        return HUGE_PENALTY
    if dist in {"student-t", "skew-student"} and not (dof > 2.0001):
        return HUGE_PENALTY
    if dist == "skew-student" and not (abs(skew) < 0.999):
        return HUGE_PENALTY
    if dist not in {"normal", "student-t", "skew-student"}:
        return HUGE_PENALTY
    denom = 1.0 - u
    h = omega / denom
    if not (h > 0.0 and math.isfinite(h)):
        return HUGE_PENALTY
    nll = 0.0
    log2pi = math.log(2.0 * K_PI)
    skew_dist = SkewStudent() if dist == "skew-student" else None
    params_vec = np.array([dof, skew], dtype=float) if dist == "skew-student" else None
    for value in r:
        eps = value - mu
        if not (h > 0.0 and math.isfinite(h)):
            return HUGE_PENALTY
        sd = math.sqrt(h)
        if dist == "normal":
            nll += 0.5 * (log2pi + math.log(h) + (eps * eps) / h)
        elif dist == "student-t":
            scale = standardized_t_scale(dof)
            z = scale * (eps / sd)
            log_fz = log_t_pdf(z, dof) + math.log(scale)
            log_feps = log_fz - 0.5 * math.log(h)
            nll -= log_feps
        else:
            log_val = skew_dist.loglikelihood(params_vec, np.array([eps]), np.array([h]))
            log_val = float(log_val) if np.isscalar(log_val) else float(log_val[0])
            nll -= log_val
        shock = eps - gamma * sd
        h = omega + alpha * (shock * shock) + beta * h
    if not math.isfinite(nll):
        return HUGE_PENALTY
    return nll


def igarch_neg_loglik(r: np.ndarray, params: Dict[str, float], dist: str, dof: float, skew: float = 0.0) -> float:
    """Evaluate the negative log-likelihood for the IGARCH model."""
    if r.size == 0:
        return HUGE_PENALTY
    alpha = params["alpha"]
    beta = params["beta"]
    mu = params["mu"]
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        return HUGE_PENALTY
    if dist in {"student-t", "skew-student"} and not (dof > 2.0001):
        return HUGE_PENALTY
    if dist == "skew-student" and not (abs(skew) < 0.999):
        return HUGE_PENALTY
    if dist not in {"normal", "student-t", "skew-student"}:
        return HUGE_PENALTY
    h = initial_variance(r, mu)
    nll = 0.0
    log2pi = math.log(2.0 * K_PI)
    skew_dist = SkewStudent() if dist == "skew-student" else None
    params_vec = np.array([dof, skew], dtype=float) if dist == "skew-student" else None
    for value in r:
        eps = value - mu
        if dist == "normal":
            nll += 0.5 * (log2pi + math.log(h) + (eps * eps) / h)
        elif dist == "student-t":
            scale = standardized_t_scale(dof)
            z = scale * (eps / math.sqrt(h))
            log_fz = log_t_pdf(z, dof) + math.log(scale)
            log_feps = log_fz - 0.5 * math.log(h)
            nll -= log_feps
        else:
            log_val = skew_dist.loglikelihood(params_vec, np.array([eps]), np.array([h]))
            log_val = float(log_val) if np.isscalar(log_val) else float(log_val[0])
            nll -= log_val
        h = alpha * (eps * eps) + beta * h
        if not (h > 0.0 and math.isfinite(h)):
            return HUGE_PENALTY
    if not math.isfinite(nll):
        return HUGE_PENALTY
    return nll


def st_neg_loglik(r: np.ndarray, params: Dict[str, float], dist: str, dof: float, skew: float = 0.0) -> float:
    """Evaluate the negative log-likelihood for the ST-GARCH model."""
    if r.size == 0:
        return HUGE_PENALTY
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    mu = params["mu"]
    shift = params["shift"]
    if not (omega > 0.0 and alpha >= 0.0 and beta >= 0.0 and gamma >= 0.0):
        return HUGE_PENALTY
    u = alpha + 0.5 * gamma + beta
    if not (u < 1.0):
        return HUGE_PENALTY
    if dist in {"student-t", "skew-student"} and not (dof > 2.0001):
        return HUGE_PENALTY
    if dist == "skew-student" and not (abs(skew) < 0.999):
        return HUGE_PENALTY
    if dist not in {"normal", "student-t", "skew-student"}:
        return HUGE_PENALTY
    denom = 1.0 - u
    h = omega / denom
    if not (h > 0.0 and math.isfinite(h)):
        return HUGE_PENALTY
    nll = 0.0
    log2pi = math.log(2.0 * K_PI)
    skew_dist = SkewStudent() if dist == "skew-student" else None
    params_vec = np.array([dof, skew], dtype=float) if dist == "skew-student" else None
    for value in r:
        eps = value - mu
        if not (h > 0.0 and math.isfinite(h)):
            return HUGE_PENALTY
        sd = math.sqrt(h)
        if dist == "normal":
            nll += 0.5 * (log2pi + math.log(h) + (eps * eps) / h)
        elif dist == "student-t":
            scale = standardized_t_scale(dof)
            z = scale * (eps / sd)
            log_fz = log_t_pdf(z, dof) + math.log(scale)
            log_feps = log_fz - 0.5 * math.log(h)
            nll -= log_feps
        else:
            log_val = skew_dist.loglikelihood(params_vec, np.array([eps]), np.array([h]))
            log_val = float(log_val) if np.isscalar(log_val) else float(log_val[0])
            nll -= log_val
        indicator = 1.0 if eps < 0.0 else 0.0
        shock = eps - shift * sd
        var_scale = alpha + gamma * indicator
        h = omega + var_scale * (shock * shock) + beta * h
    if not math.isfinite(nll):
        return HUGE_PENALTY
    return nll


def nagarch_cond_sd(r: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Compute conditional standard deviations implied by NAGARCH."""
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    mu = params["mu"]
    u = alpha * (1.0 + gamma * gamma) + beta
    denom = 1.0 - u
    h = omega / denom if denom > 1e-12 else omega
    if not (h > 0.0):
        h = 1e-6
    out = np.zeros_like(r)
    for idx, value in enumerate(r):
        sd = math.sqrt(max(h, 1e-12))
        out[idx] = sd
        eps = value - mu
        shock = eps - gamma * sd
        h_next = omega + alpha * (shock * shock) + beta * h
        if not (h_next > 0.0) or not math.isfinite(h_next):
            h_next = 1e-8
        h = h_next
    return out

def igarch_cond_sd(r: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Compute conditional standard deviations implied by IGARCH."""
    mu = params["mu"]
    alpha = params["alpha"]
    beta = params["beta"]
    h = initial_variance(r, mu)
    cond = np.empty_like(r)
    for idx, value in enumerate(r):
        cond[idx] = math.sqrt(max(h, 1e-12))
        eps = value - mu
        h = alpha * (eps * eps) + beta * h
    return cond


def st_cond_sd(r: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Compute conditional standard deviations implied by ST-GARCH."""
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    mu = params["mu"]
    shift = params["shift"]
    u = alpha + 0.5 * gamma + beta
    denom = 1.0 - u
    h = omega / denom if denom > 1e-12 else omega
    if not (h > 0.0):
        h = 1e-6
    out = np.zeros_like(r)
    for i, value in enumerate(r):
        if not (h > 0.0):
            h = 1e-6
        sd = math.sqrt(max(h, 1e-12))
        out[i] = sd
        eps = value - mu
        indicator = 1.0 if eps < 0.0 else 0.0
        shock = eps - shift * sd
        var_scale = alpha + gamma * indicator
        h_next = omega + var_scale * (shock * shock) + beta * h
        if not (h_next > 0.0) or not math.isfinite(h_next):
            h_next = 1e-8
        h = h_next
    return out

def fit_constant_vol(returns: np.ndarray) -> ModelRow:
    """Fit the constant-volatility benchmark model."""
    mu = 0.0
    omega = float(np.mean(returns ** 2))
    if not (omega > 1e-12):
        omega = 1e-12
    sd = math.sqrt(omega)
    n = returns.size
    if n == 0:
        loglik = math.nan
    else:
        loglik = -0.5 * (n * math.log(2.0 * K_PI * omega) + np.sum((returns - mu) ** 2) / omega)
    cond_sd = np.full(n, sd)
    std_resid = (returns - mu) / cond_sd
    return ModelRow(label="constant_vol", mu=mu, omega=omega, alpha=math.nan, beta=math.nan,
                    gamma=math.nan, shift=math.nan, dof=math.nan, uncond_sd=sd, loglik=loglik,
                    n_params=1, cond_sd=cond_sd, std_resid=std_resid)

def fit_arch_model(returns: np.ndarray, model: str, dist: str) -> ModelRow:
    """Fit an ARCH-family model via the arch package."""
    cm = ConstantMean(returns)
    if model == "garch":
        cm.volatility = GARCH(p=1, o=0, q=1)
    elif model == "gjr":
        cm.volatility = GARCH(p=1, o=1, q=1)
    elif model == "egarch":
        cm.volatility = EGARCH(p=1, o=1, q=1)
    else:
        raise ValueError("unknown model type")
    if dist == "normal":
        cm.distribution = Normal()
    elif dist == "student-t":
        cm.distribution = StudentsT()
    elif dist == "skew-student":
        cm.distribution = SkewStudent()
    else:
        raise ValueError("unknown dist")
    res = cm.fit(disp="off")
    params = res.params
    mu = float(params.get("mu", 0.0))
    omega = float(params.get("omega", math.nan))
    alpha = float(params.get("alpha[1]", math.nan))
    beta = float(params.get("beta[1]", math.nan))
    gamma = float(params.get("gamma[1]", math.nan)) if model == "gjr" else math.nan
    if dist == "student-t":
        dof = float(params.get("nu", math.nan))
        skew_param = math.nan
    elif dist == "skew-student":
        dof = float(params.get("eta", math.nan))
        skew_param = float(params.get("lambda", math.nan))
    else:
        dof = math.nan
        skew_param = math.nan
    if model == "garch":
        denom = 1.0 - (alpha + beta)
        uncond_sd = math.sqrt(omega / denom) if denom > 0 and omega > 0 else math.nan
    elif model == "gjr":
        gamma_val = gamma if math.isfinite(gamma) else 0.0
        denom = 1.0 - (alpha + 0.5 * gamma_val + beta)
        uncond_sd = math.sqrt(omega / denom) if denom > 0 and omega > 0 else math.nan
    else:
        uncond_sd = math.nan
    n_params = 4 if model == "garch" else 5
    if dist == "student-t":
        n_params += 1
    elif dist == "skew-student":
        n_params += 2
    cond_sd = np.asarray(getattr(res, "conditional_volatility"))
    std_resid = np.asarray(getattr(res, "std_resid"))
    label_suffix = "normal"
    if dist == "student-t":
        label_suffix = "student_t"
    elif dist == "skew-student":
        label_suffix = "skew_student_t"
    return ModelRow(label=f"{model}_{label_suffix}",
                    mu=mu, omega=omega, alpha=alpha, beta=beta, gamma=gamma,
                    shift=math.nan, skew=skew_param, dof=dof, uncond_sd=uncond_sd,
                    loglik=float(res.loglikelihood), n_params=n_params, cond_sd=cond_sd,
                    std_resid=std_resid)


def fit_nagarch_model(returns: np.ndarray, dist: str, optimizer: str = 'nelder-mead') -> ModelRow:
    """Fit the custom NAGARCH model and package results."""
    params, dof, nll = fit_nagarch(returns, dist, optimizer=optimizer)
    gamma = params["gamma"]
    persistence = params["alpha"] * (1.0 + gamma * gamma) + params["beta"]
    uncond = math.sqrt(params["omega"] / (1.0 - persistence)) if (1.0 - persistence) > 0 else math.nan
    cond_sd = nagarch_cond_sd(returns, params)
    std_resid = (returns - params["mu"]) / cond_sd
    n_params = 5 if dist == "normal" else (6 if dist == "student-t" else 7)
    if dist == "student-t":
        label = "nagarch_student_t"
        dof_val = dof
    elif dist == "skew-student":
        label = "nagarch_skew_student_t"
        dof_val = dof
    else:
        label = "nagarch_normal"
        dof_val = math.nan
    return ModelRow(label=label, mu=params["mu"], omega=params["omega"], alpha=params["alpha"],
                    beta=params["beta"], gamma=gamma, shift=math.nan, skew=params.get('skew', math.nan), dof=dof_val,
                    uncond_sd=uncond, loglik=-nll, n_params=n_params, cond_sd=cond_sd,
                    std_resid=std_resid)


def fit_igarch_model(returns: np.ndarray, dist: str, optimizer: str = 'nelder-mead') -> ModelRow:
    """Fit the custom IGARCH model and package results."""
    params, dof, nll = fit_igarch(returns, dist, optimizer=optimizer)
    cond_sd = igarch_cond_sd(returns, params)
    std_resid = (returns - params["mu"]) / cond_sd
    n_params = 2 if dist == "normal" else (3 if dist == "student-t" else 4)
    if dist == "student-t":
        label = "igarch_student_t"
        dof_val = dof
    elif dist == "skew-student":
        label = "igarch_skew_student_t"
        dof_val = dof
    else:
        label = "igarch_normal"
        dof_val = math.nan
    return ModelRow(label=label,
                    mu=params["mu"],
                    omega=math.nan,
                    alpha=params["alpha"],
                    beta=params["beta"],
                    gamma=math.nan,
                    shift=math.nan,
                    skew=params.get('skew', math.nan),
                    dof=dof_val,
                    uncond_sd=math.nan,
                    loglik=-nll,
                    n_params=n_params,
                    cond_sd=cond_sd,
                    std_resid=std_resid)


def fit_st_model(returns: np.ndarray, dist: str, optimizer: str = 'nelder-mead') -> ModelRow:
    """Fit the custom ST-GARCH model and package results."""
    params, dof, nll = fit_st(returns, dist, optimizer=optimizer)
    gamma = params["gamma"]
    shift = params["shift"]
    persistence = params["alpha"] + 0.5 * gamma + params["beta"]
    denom = 1.0 - persistence
    uncond = math.sqrt(params["omega"] / denom) if denom > 0 and params["omega"] > 0 else math.nan
    cond_sd = st_cond_sd(returns, params)
    std_resid = (returns - params["mu"]) / cond_sd
    n_params = 6 if dist == "normal" else (7 if dist == "student-t" else 8)
    if dist == "student-t":
        label = "st_student_t"
        dof_val = dof
    elif dist == "skew-student":
        label = "st_skew_student_t"
        dof_val = dof
    else:
        label = "st_normal"
        dof_val = math.nan
    return ModelRow(
        label=label,
        mu=params["mu"],
        omega=params["omega"],
        alpha=params["alpha"],
        beta=params["beta"],
        gamma=gamma,
        shift=shift,
        skew=params.get('skew', math.nan),
        dof=dof_val,
        uncond_sd=uncond,
        loglik=-nll,
        n_params=n_params,
        cond_sd=cond_sd,
        std_resid=std_resid,
    )


def compute_loglik_select_values(rows: List[ModelRow]) -> List[float]:
    """Return negated log-likelihoods for ranking purposes."""
    out = []
    for row in rows:
        out.append(-row.loglik if math.isfinite(row.loglik) else math.inf)
    return out

def compute_metric_order(rows: List[ModelRow], metric_values: List[float]) -> List[str]:
    """Return model labels ordered by a selection metric."""
    indices = sorted_indices(metric_values, ascending=True)
    return [rows[idx].label for idx in indices]
