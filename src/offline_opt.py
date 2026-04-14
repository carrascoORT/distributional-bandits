import numpy as np
from scipy.optimize import minimize


def project_to_simplex_check(w, tol=1e-8):
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or np.any(w < -tol) or not np.isclose(np.sum(w), 1.0, atol=tol):
        return False
    return True


def solve_utility_optimum_gamma(instance, utility, gamma, x0=None):
    K = instance.K
    if not (0.0 <= gamma < 1.0 / K):
        raise ValueError("gamma must satisfy 0 <= gamma < 1/K.")
    if x0 is None:
        x0 = np.ones(K, dtype=float) / K
    else:
        x0 = np.asarray(x0, dtype=float)

    def objective(w):
        return -utility.value(instance, w)

    jac = None
    if hasattr(utility, "gradient"):
        def gradient(w):
            return -utility.gradient(instance, w)
        jac = gradient

    constraints = [{
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
        "jac": lambda w: np.ones_like(w),
    }]
    bounds = [(gamma, 1.0) for _ in range(K)]

    result = minimize(
        fun=objective,
        x0=x0,
        jac=jac,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
    )

    w_star = np.asarray(result.x, dtype=float)
    w_star[w_star < 0] = 0.0
    s = np.sum(w_star)
    if s <= 0:
        raise RuntimeError("Optimization failed: nonpositive sum of weights.")
    w_star = w_star / s
    if not project_to_simplex_check(w_star):
        raise RuntimeError("Optimization returned a point outside the simplex.")
    u_star = float(utility.value(instance, w_star))
    return w_star, u_star, result


def solve_variance_optimum(instance, utility, x0=None):
    return solve_utility_optimum_gamma(instance, utility, gamma=0.0, x0=x0)


def solve_variance_optimum_gamma(instance, utility, gamma, x0=None):
    return solve_utility_optimum_gamma(instance, utility, gamma=gamma, x0=x0)
