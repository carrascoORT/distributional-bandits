import numpy as np


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
        """
        Reset the internal state of the algorithm.
        """
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
    Useful as a baseline and for debugging.
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
        # No learning: weights stay fixed.
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


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    """
    x = np.asarray(x, dtype=float)
    x_shift = x - np.max(x)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x)


class SimpleMirrorAscent(BanditAlgorithm):
    """
    Very simple mirror-ascent-style skeleton.

    It maintains logits h and corresponding softmax weights w.
    The update rule currently uses a simple placeholder reward signal.
    """

    def __init__(self, K: int, eta: float = 0.1):
        super().__init__(K=K, name="simple_mirror_ascent")
        if eta <= 0:
            raise ValueError("eta must be positive.")
        self.eta = float(eta)
        self.h = np.zeros(K, dtype=float)
        self.w = np.ones(K, dtype=float) / K

    def reset(self, rng: np.random.Generator):
        super().reset(rng)
        self.h = np.zeros(self.K, dtype=float)
        self.w = np.ones(self.K, dtype=float) / self.K

    def select_action(self) -> int:
        return int(self.rng.choice(self.K, p=self.w))

    def update(self, action: int, reward: float):
        """
        Placeholder update.

        We use the score
            g_k = (1_{A_t = k} - w_k) * reward
        which has the right 'logit-gradient-like' shape.
        """
        indicator = np.zeros(self.K, dtype=float)
        indicator[action] = 1.0

        g = (indicator - self.w) * reward
        self.h = self.h + self.eta * g
        self.w = softmax(self.h)

    def current_weights(self) -> np.ndarray:
        return self.w.copy()