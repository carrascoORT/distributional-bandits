import numpy as np
from scipy import stats


class Arm:
    """Base class for a one-dimensional reward distribution."""

    def sample(self, rng: np.random.Generator) -> float:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def second_moment(self) -> float:
        raise NotImplementedError

    def variance(self) -> float:
        m = self.mean()
        return self.second_moment() - m**2

    def cdf(self, x):
        raise NotImplementedError

    def ppf(self, u):
        raise NotImplementedError

    def pdf(self, x):
        raise NotImplementedError

    def support_bounds(self, eps: float = 1e-4):
        """Finite plotting / numerical bounds, possibly using extreme quantiles."""
        lo = float(np.asarray(self.ppf(eps)).reshape(-1)[0])
        hi = float(np.asarray(self.ppf(1.0 - eps)).reshape(-1)[0])
        return lo, hi


class UniformArm(Arm):
    def __init__(self, a: float, b: float):
        if b <= a:
            raise ValueError("Require b > a for UniformArm.")
        self.a = float(a)
        self.b = float(b)
        self._dist = stats.uniform(loc=self.a, scale=self.b - self.a)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.a, self.b))

    def mean(self) -> float:
        return 0.5 * (self.a + self.b)

    def second_moment(self) -> float:
        return (self.a**2 + self.a * self.b + self.b**2) / 3.0

    def cdf(self, x):
        return self._dist.cdf(x)

    def ppf(self, u):
        return self._dist.ppf(u)

    def pdf(self, x):
        return self._dist.pdf(x)

    def support_bounds(self, eps: float = 1e-4):
        return self.a, self.b

    def __repr__(self) -> str:
        return f"UniformArm(a={self.a}, b={self.b})"


class BetaArm(Arm):
    """Beta(alpha, beta) distribution rescaled from [0,1] to [a,b]."""

    def __init__(self, alpha: float, beta: float, a: float = 0.0, b: float = 1.0):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive.")
        if b <= a:
            raise ValueError("Require b > a for BetaArm.")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.a = float(a)
        self.b = float(b)
        self._base = stats.beta(self.alpha, self.beta)
        self._scale = self.b - self.a

    def sample(self, rng: np.random.Generator) -> float:
        x = rng.beta(self.alpha, self.beta)
        return float(self.a + self._scale * x)

    def _mean_unit(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def _second_moment_unit(self) -> float:
        a = self.alpha
        b = self.beta
        return a * (a + 1.0) / ((a + b) * (a + b + 1.0))

    def mean(self) -> float:
        return self.a + self._scale * self._mean_unit()

    def second_moment(self) -> float:
        c = self.a
        d = self._scale
        ex = self._mean_unit()
        ex2 = self._second_moment_unit()
        return c**2 + 2.0 * c * d * ex + d**2 * ex2

    def cdf(self, x):
        z = (np.asarray(x) - self.a) / self._scale
        vals = self._base.cdf(z)
        vals = np.where(z <= 0.0, 0.0, vals)
        vals = np.where(z >= 1.0, 1.0, vals)
        return vals

    def ppf(self, u):
        return self.a + self._scale * self._base.ppf(u)

    def pdf(self, x):
        z = (np.asarray(x) - self.a) / self._scale
        vals = self._base.pdf(z) / self._scale
        vals = np.where((z < 0.0) | (z > 1.0), 0.0, vals)
        return vals

    def support_bounds(self, eps: float = 1e-4):
        return self.a, self.b

    def __repr__(self) -> str:
        return f"BetaArm(alpha={self.alpha}, beta={self.beta}, a={self.a}, b={self.b})"


class TriangularArm(Arm):
    def __init__(self, left: float, mode: float, right: float):
        if not (left <= mode <= right):
            raise ValueError("Require left <= mode <= right.")
        if right <= left:
            raise ValueError("Require right > left for TriangularArm.")

        self.left = float(left)
        self.mode = float(mode)
        self.right = float(right)
        c = (self.mode - self.left) / (self.right - self.left)
        self._dist = stats.triang(c=c, loc=self.left, scale=self.right - self.left)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.triangular(self.left, self.mode, self.right))

    def mean(self) -> float:
        return (self.left + self.mode + self.right) / 3.0

    def variance(self) -> float:
        l = self.left
        m = self.mode
        r = self.right
        return (l**2 + m**2 + r**2 - l * m - l * r - m * r) / 18.0

    def second_moment(self) -> float:
        mu = self.mean()
        return self.variance() + mu**2

    def cdf(self, x):
        return self._dist.cdf(x)

    def ppf(self, u):
        return self._dist.ppf(u)

    def pdf(self, x):
        return self._dist.pdf(x)

    def support_bounds(self, eps: float = 1e-4):
        return self.left, self.right

    def __repr__(self) -> str:
        return f"TriangularArm(left={self.left}, mode={self.mode}, right={self.right})"


class GaussianArm(Arm):
    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be positive for GaussianArm.")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self._dist = stats.norm(loc=self.mu, scale=self.sigma)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mu, self.sigma))

    def mean(self) -> float:
        return self.mu

    def second_moment(self) -> float:
        return self.mu**2 + self.sigma**2

    def cdf(self, x):
        return self._dist.cdf(x)

    def ppf(self, u):
        return self._dist.ppf(u)

    def pdf(self, x):
        return self._dist.pdf(x)

    def __repr__(self) -> str:
        return f"GaussianArm(mu={self.mu}, sigma={self.sigma})"
