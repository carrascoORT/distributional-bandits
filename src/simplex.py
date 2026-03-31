import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    """
    x = np.asarray(x, dtype=float)
    x_shift = x - np.max(x)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x)


def kl_project_to_truncated_simplex(p: np.ndarray, gamma: float) -> np.ndarray:
    """
    KL projection of a probability vector p onto

        Delta_gamma = {w : w_i >= gamma, sum_i w_i = 1}.

    More precisely, this returns the minimizer of
        sum_i w_i log(w_i / p_i)
    over w in Delta_gamma.

    Assumes p_i > 0 for all i.
    """
    p = np.asarray(p, dtype=float)
    K = len(p)

    if np.any(p <= 0):
        raise ValueError("All entries of p must be strictly positive.")
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("p must sum to 1.")
    if not (0.0 <= gamma < 1.0 / K):
        raise ValueError("gamma must satisfy 0 <= gamma < 1/K.")

    # If already feasible, return p
    if np.all(p >= gamma):
        return p.copy()

    # We seek c > 0 such that sum_i max(gamma, c p_i) = 1.
    # The left-hand side is increasing in c, so use bisection.
    def f(c):
        return np.sum(np.maximum(gamma, c * p)) - 1.0

    c_low = 0.0
    c_high = 1.0

    while f(c_high) < 0:
        c_high *= 2.0

    for _ in range(100):
        c_mid = 0.5 * (c_low + c_high)
        if f(c_mid) < 0:
            c_low = c_mid
        else:
            c_high = c_mid

    c_star = c_high
    w = np.maximum(gamma, c_star * p)
    w = w / np.sum(w)

    return w