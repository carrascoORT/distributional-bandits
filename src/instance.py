import numpy as np


class BanditInstance:
    """
    A stochastic bandit instance given by a list of arm objects.

    Each arm is expected to implement:
    - sample(rng)
    - mean()
    - second_moment()
    - variance()
    """

    def __init__(self, arms, name: str = "bandit_instance"):
        if len(arms) == 0:
            raise ValueError("A bandit instance must contain at least one arm.")
        self.arms = list(arms)
        self.K = len(self.arms)
        self.name = name

    def sample(self, arm_index: int, rng: np.random.Generator) -> float:
        if not (0 <= arm_index < self.K):
            raise IndexError(f"arm_index must be in {{0, ..., {self.K - 1}}}.")
        return self.arms[arm_index].sample(rng)

    def means(self) -> np.ndarray:
        return np.array([arm.mean() for arm in self.arms], dtype=float)

    def second_moments(self) -> np.ndarray:
        return np.array([arm.second_moment() for arm in self.arms], dtype=float)

    def variances(self) -> np.ndarray:
        return np.array([arm.variance() for arm in self.arms], dtype=float)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "K": self.K,
            "means": self.means(),
            "second_moments": self.second_moments(),
            "variances": self.variances(),
        }

    def __repr__(self) -> str:
        return f"BanditInstance(name={self.name!r}, K={self.K})"