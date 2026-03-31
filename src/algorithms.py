import numpy as np

from src.simplex import softmax, kl_project_to_truncated_simplex


class BanditAlgorithm:
    """
    Base class for algorithms on a K-armed bandit.
    """

    def __init__(self, K: int, name: str = "algorithm"):
        if K <= 0:
            raise ValueError("K must be a positive integer.")
        self.K = int(K)
        self.name = name
        self.rng = None

    def reset(self, rng: np.random.Generator):
        self.rng = rng

    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, action: int, reward: float):
        raise NotImplementedError

    def current_weights(self) -> np.ndarray:
        raise NotImplementedError


class FixedMixturePolicy(BanditAlgorithm):
    """
    Policy that samples actions i.i.d. from a fixed weight vector.
    """

    def __init__(self, weights, name: str = "fixed_mixture"):
        weights = np.asarray(weights, dtype=float)

        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array.")
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative.")
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("weights must sum to 1.")

        super().__init__(K=len(weights), name=name)
        self.weights = weights.copy()

    def reset(self, rng: np.random.Generator):
        super().reset(rng)

    def select_action(self) -> int:
        return int(self.rng.choice(self.K, p=self.weights))

    def update(self, action: int, reward: float):
        pass

    def current_weights(self) -> np.ndarray:
        return self.weights.copy()


class UniformPolicy(FixedMixturePolicy):
    """
    Uniform mixture over all arms.
    """

    def __init__(self, K: int):
        weights = np.ones(K, dtype=float) / K
        super().__init__(weights=weights, name="uniform")


class VarianceMirrorAscent(BanditAlgorithm):
    """
    Mirror-ascent algorithm for the variance utility using exact arm moments.
    """

    def __init__(self, K: int, eta0: float = 0.2, gamma: float = 0.0):
        super().__init__(K=K, name="variance_mirror_ascent")

        if eta0 <= 0:
            raise ValueError("eta0 must be positive.")
        if not (0.0 <= gamma < 1.0 / K):
            raise ValueError("gamma must satisfy 0 <= gamma < 1/K.")

        self.eta0 = float(eta0)
        self.gamma = float(gamma)

        self.h = np.zeros(K, dtype=float)
        self.w = np.ones(K, dtype=float) / K
        self.t = 0

        self.instance = None

    def reset(self, rng: np.random.Generator, instance=None):
        super().reset(rng)
        self.h = np.zeros(self.K, dtype=float)
        self.w = np.ones(self.K, dtype=float) / self.K
        self.t = 0

        if instance is not None:
            self.instance = instance

        if self.instance is None:
            raise ValueError("VarianceMirrorAscent.reset requires instance=...")

        if self.gamma > 0.0:
            self.w = kl_project_to_truncated_simplex(self.w, self.gamma)

    def select_action(self) -> int:
        return int(self.rng.choice(self.K, p=self.w))

    def step_size(self) -> float:
        return self.eta0 / np.sqrt(self.t + 1)

    def current_weights(self) -> np.ndarray:
        return self.w.copy()

    def update(self, action: int, reward: float):
        mu = self.instance.means()
        m2 = self.instance.second_moments()

        mu_w = float(np.dot(self.w, mu))
        u_w = float(np.dot(self.w, m2) - mu_w**2)

        phi = (reward - mu_w) ** 2 - u_w

        e_a = np.zeros(self.K, dtype=float)
        e_a[action] = 1.0

        g = (e_a - self.w) * phi

        eta_t = self.step_size()
        self.h = self.h + eta_t * g

        p = softmax(self.h)

        if self.gamma > 0.0:
            self.w = kl_project_to_truncated_simplex(p, self.gamma)
        else:
            self.w = p

        self.t += 1


class VarianceIFAscent(BanditAlgorithm):
    """
    Mirror-ascent algorithm for the variance utility using a plug-in
    influence function built from empirical arm distributions.

    For each arm k, we keep:
      - N_k       = number of samples
      - S1_k      = sum of rewards
      - S2_k      = sum of squared rewards

    The empirical mixture moments at weights w are:
      hat mu_w   = sum_k w_k hat mu_k
      hat m2_w   = sum_k w_k hat m2_k
      hat U(w)   = hat m2_w - hat mu_w^2

    and the plug-in IF is:
      hat phi_w(r) = (r - hat mu_w)^2 - hat U(w).
    """

    def __init__(
        self,
        K: int,
        eta0: float = 0.2,
        gamma: float = 0.0,
        prior_mean: float = 0.0,
        prior_second_moment: float = 1.0,
        prior_count: float = 1.0,
    ):
        super().__init__(K=K, name="variance_if_ascent")

        if eta0 <= 0:
            raise ValueError("eta0 must be positive.")
        if not (0.0 <= gamma < 1.0 / K):
            raise ValueError("gamma must satisfy 0 <= gamma < 1/K.")
        if prior_count < 0:
            raise ValueError("prior_count must be nonnegative.")

        self.eta0 = float(eta0)
        self.gamma = float(gamma)

        self.prior_mean = float(prior_mean)
        self.prior_second_moment = float(prior_second_moment)
        self.prior_count = float(prior_count)

        self.h = np.zeros(K, dtype=float)
        self.w = np.ones(K, dtype=float) / K
        self.t = 0

        self.counts = np.zeros(K, dtype=float)
        self.sum_rewards = np.zeros(K, dtype=float)
        self.sum_sq_rewards = np.zeros(K, dtype=float)

    def reset(self, rng: np.random.Generator, instance=None):
        super().reset(rng)
        self.h = np.zeros(self.K, dtype=float)
        self.w = np.ones(self.K, dtype=float) / self.K
        self.t = 0

        self.counts = np.zeros(self.K, dtype=float)
        self.sum_rewards = np.zeros(self.K, dtype=float)
        self.sum_sq_rewards = np.zeros(self.K, dtype=float)

        if self.gamma > 0.0:
            self.w = kl_project_to_truncated_simplex(self.w, self.gamma)

    def select_action(self) -> int:
        return int(self.rng.choice(self.K, p=self.w))

    def step_size(self) -> float:
        return self.eta0 / np.sqrt(self.t + 1)

    def current_weights(self) -> np.ndarray:
        return self.w.copy()

    def empirical_means(self) -> np.ndarray:
        denom = self.counts + self.prior_count
        numer = self.sum_rewards + self.prior_count * self.prior_mean
        return numer / denom

    def empirical_second_moments(self) -> np.ndarray:
        denom = self.counts + self.prior_count
        numer = self.sum_sq_rewards + self.prior_count * self.prior_second_moment
        return numer / denom

    def plugin_mixture_mean(self) -> float:
        mu_hat = self.empirical_means()
        return float(np.dot(self.w, mu_hat))

    def plugin_mixture_second_moment(self) -> float:
        m2_hat = self.empirical_second_moments()
        return float(np.dot(self.w, m2_hat))

    def plugin_variance(self) -> float:
        mu_w_hat = self.plugin_mixture_mean()
        m2_w_hat = self.plugin_mixture_second_moment()
        return float(m2_w_hat - mu_w_hat**2)

    def update(self, action: int, reward: float):
        """
        Use the plug-in IF built from the current empirical arm laws BEFORE
        incorporating the new reward. Then update the empirical statistics.
        """

        mu_hat = self.empirical_means()
        m2_hat = self.empirical_second_moments()

        mu_w_hat = float(np.dot(self.w, mu_hat))
        u_w_hat = float(np.dot(self.w, m2_hat) - mu_w_hat**2)

        phi_hat = (reward - mu_w_hat) ** 2 - u_w_hat

        e_a = np.zeros(self.K, dtype=float)
        e_a[action] = 1.0

        g = (e_a - self.w) * phi_hat

        eta_t = self.step_size()
        self.h = self.h + eta_t * g

        p = softmax(self.h)
        if self.gamma > 0.0:
            self.w = kl_project_to_truncated_simplex(p, self.gamma)
        else:
            self.w = p

        # Update empirical arm statistics AFTER using phi_hat
        self.counts[action] += 1.0
        self.sum_rewards[action] += reward
        self.sum_sq_rewards[action] += reward**2

        self.t += 1