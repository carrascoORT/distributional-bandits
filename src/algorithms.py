import numpy as np

from src.simplex import softmax, kl_project_to_truncated_simplex


class BanditAlgorithm:
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

    def latest_diagnostics(self) -> dict:
        return {}


class FixedMixturePolicy(BanditAlgorithm):
    def __init__(self, weights, name: str = "fixed_mixture"):
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array.")
        if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("weights must be nonnegative and sum to 1.")
        super().__init__(K=len(weights), name=name)
        self.weights = weights.copy()

    def select_action(self) -> int:
        return int(self.rng.choice(self.K, p=self.weights))

    def update(self, action: int, reward: float):
        return None

    def current_weights(self) -> np.ndarray:
        return self.weights.copy()


class UniformPolicy(FixedMixturePolicy):
    def __init__(self, K: int):
        super().__init__(weights=np.ones(K, dtype=float) / K, name="uniform")


class VarianceMirrorAscent(BanditAlgorithm):
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

    def reset(self, rng: np.random.Generator, instance=None, utility=None):
        super().reset(rng)
        self.h[:] = 0.0
        self.w[:] = 1.0 / self.K
        self.t = 0
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
        self.h = self.h + self.step_size() * g
        p = softmax(self.h)
        self.w = kl_project_to_truncated_simplex(p, self.gamma) if self.gamma > 0.0 else p
        self.t += 1


class VarianceIFAscent(BanditAlgorithm):
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

    def reset(self, rng: np.random.Generator, instance=None, utility=None):
        super().reset(rng)
        self.h[:] = 0.0
        self.w[:] = 1.0 / self.K
        self.t = 0
        self.counts[:] = 0.0
        self.sum_rewards[:] = 0.0
        self.sum_sq_rewards[:] = 0.0
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

    def update(self, action: int, reward: float):
        mu_hat = self.empirical_means()
        m2_hat = self.empirical_second_moments()
        mu_w_hat = float(np.dot(self.w, mu_hat))
        u_w_hat = float(np.dot(self.w, m2_hat) - mu_w_hat**2)
        phi_hat = (reward - mu_w_hat) ** 2 - u_w_hat
        e_a = np.zeros(self.K, dtype=float)
        e_a[action] = 1.0
        g = (e_a - self.w) * phi_hat
        self.h = self.h + self.step_size() * g
        p = softmax(self.h)
        self.w = kl_project_to_truncated_simplex(p, self.gamma) if self.gamma > 0.0 else p
        self.counts[action] += 1.0
        self.sum_rewards[action] += reward
        self.sum_sq_rewards[action] += reward**2
        self.t += 1


class WassersteinMirrorAscent(BanditAlgorithm):
    def __init__(self, K: int, eta0: float = 0.2, gamma: float = 0.0):
        super().__init__(K=K, name="wasserstein_mirror_ascent")
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
        self.utility = None

    def reset(self, rng: np.random.Generator, instance=None, utility=None):
        super().reset(rng)
        self.h[:] = 0.0
        self.w[:] = 1.0 / self.K
        self.t = 0
        self.instance = instance
        self.utility = utility
        if self.instance is None or self.utility is None:
            raise ValueError("WassersteinMirrorAscent.reset requires instance=... and utility=...")
        if self.gamma > 0.0:
            self.w = kl_project_to_truncated_simplex(self.w, self.gamma)

    def select_action(self) -> int:
        return int(self.rng.choice(self.K, p=self.w))

    def step_size(self) -> float:
        return self.eta0 / np.sqrt(self.t + 1)

    def current_weights(self) -> np.ndarray:
        return self.w.copy()

    def update(self, action: int, reward: float):
        g = self.utility.exact_gradient_coordinates(self.instance, self.w)
        self.h = self.h + self.step_size() * g
        p = softmax(self.h)
        self.w = kl_project_to_truncated_simplex(p, self.gamma) if self.gamma > 0.0 else p
        self.t += 1


class WassersteinIFAscent(BanditAlgorithm):
    r"""
    Stochastic IF ascent for U(P) = -W_2(P,Q)^2.

    For mixture_source='exact', the IF is evaluated at the exact mixture law.
    For mixture_source='empirical', the IF is evaluated at the plug-in empirical mixture.
    """

    def __init__(
        self,
        K: int,
        eta0: float = 0.2,
        gamma: float = 0.0,
        prior_count: float = 1.0,
        mixture_source: str = "empirical",
    ):
        mixture_source = str(mixture_source).lower()
        if mixture_source not in {"exact", "empirical"}:
            raise ValueError("mixture_source must be 'exact' or 'empirical'.")

        algo_name = (
            "wasserstein_if_ascent_exact"
            if mixture_source == "exact"
            else "wasserstein_if_ascent_empirical"
        )
        super().__init__(K=K, name=algo_name)

        if eta0 <= 0:
            raise ValueError("eta0 must be positive.")
        if not (0.0 <= gamma < 1.0 / K):
            raise ValueError("gamma must satisfy 0 <= gamma < 1/K.")
        if prior_count < 0:
            raise ValueError("prior_count must be nonnegative.")

        self.eta0 = float(eta0)
        self.gamma = float(gamma)
        self.prior_count = float(prior_count)
        self.mixture_source = mixture_source

        self.h = np.zeros(K, dtype=float)
        self.w = np.ones(K, dtype=float) / K
        self.t = 0
        self.instance = None
        self.utility = None
        self.counts = np.zeros(K, dtype=float)
        self.cdf_counts = None
        self.prior_cdf = None
        self.x_grid = None

    def reset(self, rng: np.random.Generator, instance=None, utility=None):
        super().reset(rng)
        self.h[:] = 0.0
        self.w[:] = 1.0 / self.K
        self.t = 0
        self.instance = instance
        self.utility = utility
        if self.instance is None or self.utility is None:
            raise ValueError("WassersteinIFAscent.reset requires instance=... and utility=...")
        if self.gamma > 0.0:
            self.w = kl_project_to_truncated_simplex(self.w, self.gamma)

        self.x_grid = self.utility._prepare_cache(self.instance)["x_grid"]
        m = len(self.x_grid)
        self.counts = np.zeros(self.K, dtype=float)
        self.cdf_counts = np.zeros((self.K, m), dtype=float)
        self.prior_cdf = np.tile(self.utility.target_cdf_grid(self.instance), (self.K, 1))

    def select_action(self) -> int:
        return int(self.rng.choice(self.K, p=self.w))

    def step_size(self) -> float:
        return self.eta0 / np.sqrt(self.t + 1)

    def current_weights(self) -> np.ndarray:
        return self.w.copy()

    def _empirical_arm_cdf_grid(self):
        denom = self.counts[:, None] + self.prior_count
        numer = self.cdf_counts + self.prior_count * self.prior_cdf
        return numer / denom

    def _current_if_data(self):
        if self.mixture_source == "exact":
            return self.utility.exact_influence_function(self.instance, self.w)

        arm_cdf_grid = self._empirical_arm_cdf_grid()
        mix_cdf = np.dot(self.w, arm_cdf_grid)
        return self.utility.influence_function_from_cdf(
            self.instance,
            mix_cdf,
            center_mode="mixture",
        )

    def _update_empirical_state(self, action: int, reward: float):
        idx = int(np.searchsorted(self.x_grid, reward, side="left"))
        idx = max(0, min(idx, len(self.x_grid) - 1))
        self.cdf_counts[action, idx:] += 1.0
        self.counts[action] += 1.0

    def _single_gradient_estimate(self, action: int, reward: float, if_data) -> np.ndarray:
        psi = float(self.utility.evaluate_if_on_rewards(if_data, np.array([reward]))[0])
        e_a = np.zeros(self.K, dtype=float)
        e_a[action] = 1.0
        return ((e_a / self.w) - 1.0) * psi

    def estimate_conditional_bias_mc(
        self,
        rng: np.random.Generator,
        n_mc: int = 256,
    ) -> dict:
        """
        Monte Carlo estimate of
            B_t = E[hat g_t | F_{t-1}] - g_t
        at the current pre-update state.

        This is intended for the plug-in IF method (mixture_source='empirical').
        """
        if self.instance is None or self.utility is None:
            raise ValueError("estimate_conditional_bias_mc requires instance and utility to be set.")
        if n_mc <= 0:
            raise ValueError("n_mc must be positive.")

        if_data = self._current_if_data()
        g_exact = self.utility.exact_gradient_coordinates(self.instance, self.w)

        ghat_sum = np.zeros(self.K, dtype=float)
        for _ in range(n_mc):
            action = int(rng.choice(self.K, p=self.w))
            reward = float(self.instance.sample(action, rng))
            ghat = self._single_gradient_estimate(action, reward, if_data)
            ghat_sum += ghat

        ghat_mean = ghat_sum / float(n_mc)
        bias = ghat_mean - g_exact

        return {
            "mc_gradient_mean": ghat_mean,
            "exact_gradient": g_exact,
            "mc_bias_vector": bias,
            "mc_bias_inf": float(np.max(np.abs(bias))),
            "mc_bias_l2": float(np.linalg.norm(bias)),
        }

    def update(self, action: int, reward: float):
        if_data = self._current_if_data()
        ghat = self._single_gradient_estimate(action, reward, if_data)

        self.h = self.h + self.step_size() * ghat
        p = softmax(self.h)
        self.w = kl_project_to_truncated_simplex(p, self.gamma) if self.gamma > 0.0 else p
        self._update_empirical_state(action, reward)
        self.t += 1