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
        return (self.a**2 + self.a * self.b + self.b**2) / 3.0

    def __repr__(self) -> str:
        return f"UniformArm(a={self.a}, b={self.b})"


class BetaArm(Arm):
    """
    Beta(alpha, beta) distribution rescaled from [0,1] to [a,b].

    If X ~ Beta(alpha, beta), then R = a + (b-a) X.
    """

    def __init__(self, alpha: float, beta: float, a: float = 0.0, b: float = 1.0):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive.")
        if b <= a:
            raise ValueError("Require b > a for BetaArm.")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.a = float(a)
        self.b = float(b)

    def sample(self, rng: np.random.Generator) -> float:
        x = rng.beta(self.alpha, self.beta)
        return float(self.a + (self.b - self.a) * x)

    def _mean_unit(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def _second_moment_unit(self) -> float:
        a = self.alpha
        b = self.beta
        return a * (a + 1.0) / ((a + b) * (a + b + 1.0))

    def mean(self) -> float:
        m = self._mean_unit()
        return self.a + (self.b - self.a) * m

    def second_moment(self) -> float:
        """
        For R = c + dX,
            E[R^2] = c^2 + 2cd E[X] + d^2 E[X^2].
        """
        c = self.a
        d = self.b - self.a
        ex = self._mean_unit()
        ex2 = self._second_moment_unit()
        return c**2 + 2.0 * c * d * ex + d**2 * ex2

    def __repr__(self) -> str:
        return (
            f"BetaArm(alpha={self.alpha}, beta={self.beta}, "
            f"a={self.a}, b={self.b})"
        )


class TriangularArm(Arm):
    """
    Triangular distribution on [left, right] with mode.

    Uses NumPy's triangular sampler.
    """

    def __init__(self, left: float, mode: float, right: float):
        if not (left <= mode <= right):
            raise ValueError("Require left <= mode <= right.")
        if right <= left:
            raise ValueError("Require right > left for TriangularArm.")

        self.left = float(left)
        self.mode = float(mode)
        self.right = float(right)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.triangular(self.left, self.mode, self.right))

    def mean(self) -> float:
        return (self.left + self.mode + self.right) / 3.0

    def variance(self) -> float:
        l = self.left
        m = self.mode
        r = self.right
        return (
            l**2 + m**2 + r**2 - l * m - l * r - m * r
        ) / 18.0

    def second_moment(self) -> float:
        mu = self.mean()
        var = self.variance()
        return var + mu**2

    def __repr__(self) -> str:
        return (
            f"TriangularArm(left={self.left}, mode={self.mode}, right={self.right})"
        )