import numpy as np


class Arm:
    """
    Base class for a bandit arm.

    Subclasses should implement:
    - sample(rng): draw one reward
    - mean(): exact mean
    - second_moment(): exact second moment E[R^2]
    """

    def sample(self, rng: np.random.Generator) -> float:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def second_moment(self) -> float:
        raise NotImplementedError

    def variance(self) -> float:
        m = self.mean()
        return self.second_moment() - m**2


class UniformArm(Arm):
    """
    Uniform distribution on [a, b].
    """

    def __init__(self, a: float, b: float):
        if b <= a:
            raise ValueError("Require b > a for UniformArm.")
        self.a = float(a)
        self.b = float(b)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.a, self.b))

    def mean(self) -> float:
        return 0.5 * (self.a + self.b)

    def second_moment(self) -> float:
        # E[X^2] for X ~ Uniform[a,b]
        return (self.a**2 + self.a * self.b + self.b**2) / 3.0

    def __repr__(self) -> str:
        return f"UniformArm(a={self.a}, b={self.b})"