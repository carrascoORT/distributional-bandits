import numpy as np


class VarianceUtility:
    """
    Variance utility for mixture weights w on a bandit instance.

    If the arm means are mu_k and second moments are m2_k, then
        U(w) = sum_k w_k m2_k - (sum_k w_k mu_k)^2.
    """

    def value(self, instance, w) -> float:
        w = np.asarray(w, dtype=float)

        if w.ndim != 1:
            raise ValueError("w must be a one-dimensional array.")
        if len(w) != instance.K:
            raise ValueError("w must have length instance.K.")
        if np.any(w < -1e-12):
            raise ValueError("w must have nonnegative entries.")
        if not np.isclose(np.sum(w), 1.0):
            raise ValueError("w must sum to 1.")

        mu = instance.means()
        m2 = instance.second_moments()

        return float(np.dot(w, m2) - np.dot(w, mu) ** 2)

    def gradient(self, instance, w) -> np.ndarray:
        """
        Gradient of U(w) on R^K:
            grad U(w) = m2 - 2 (mu^T w) mu
        """
        w = np.asarray(w, dtype=float)
        mu = instance.means()
        m2 = instance.second_moments()

        return m2 - 2.0 * np.dot(mu, w) * mu

    def hessian(self, instance) -> np.ndarray:
        """
        Hessian of U(w):
            Hess U(w) = -2 mu mu^T
        """
        mu = instance.means()
        return -2.0 * np.outer(mu, mu)