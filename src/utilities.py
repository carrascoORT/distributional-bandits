import numpy as np

from src.instance_factory import build_distribution_from_spec


class VarianceUtility:
    def value(self, instance, w) -> float:
        w = np.asarray(w, dtype=float)
        if w.ndim != 1 or len(w) != instance.K:
            raise ValueError("w must be a one-dimensional array of length instance.K.")
        mu = instance.means()
        m2 = instance.second_moments()
        return float(np.dot(w, m2) - np.dot(w, mu) ** 2)

    def gradient(self, instance, w) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        mu = instance.means()
        m2 = instance.second_moments()
        return m2 - 2.0 * np.dot(mu, w) * mu

    def hessian(self, instance) -> np.ndarray:
        mu = instance.means()
        return -2.0 * np.outer(mu, mu)


class WassersteinUtility:
    """
    One-dimensional utility U(P) = -W_2(P, Q)^2.

    This class supports two closely related computations:

    1) exact utility values for synthetic bandit instances, using exact arm CDFs;
    2) numerical influence-function approximations on an x-grid, either at the
       exact mixture law or at a plug-in empirical mixture law.

    The target inverse CDF F_Q^{-1} can be obtained either from a closed-form
    formula supplied by the target distribution, or from a precomputed numeric
    inversion of the target CDF on the working grid.
    """

    def __init__(
        self,
        target_dist,
        n_grid: int = 401,
        quantile_eps: float = 1e-4,
        target_quantile_mode: str = "auto",
        target_quantile_grid_size: int | None = None,
    ):
        if n_grid < 50:
            raise ValueError("n_grid must be at least 50.")
        if not (0.0 < quantile_eps < 0.1):
            raise ValueError("quantile_eps must lie in (0, 0.1).")
        target_quantile_mode = str(target_quantile_mode).lower()
        if target_quantile_mode not in {"auto", "formula", "numerical"}:
            raise ValueError("target_quantile_mode must be one of {'auto','formula','numerical'}." )
        self.target_dist = target_dist
        self.n_grid = int(n_grid)
        self.quantile_eps = float(quantile_eps)
        self.target_quantile_mode = target_quantile_mode
        self.target_quantile_grid_size = int(target_quantile_grid_size or max(4 * self.n_grid, 2001))
        self._cache = {}

    @classmethod
    def from_config(cls, utility_cfg: dict):
        target_cfg = utility_cfg.get("target", None)
        if target_cfg is None:
            raise ValueError("wasserstein utility requires a target distribution specification.")
        target_dist = build_distribution_from_spec(target_cfg)
        n_grid = int(utility_cfg.get("n_grid", 401))
        quantile_eps = float(utility_cfg.get("quantile_eps", 1e-4))
        tq_cfg = utility_cfg.get("target_quantile", {}) or {}
        mode = tq_cfg.get("mode", utility_cfg.get("target_quantile_mode", "auto"))
        grid_size = tq_cfg.get("grid_size", utility_cfg.get("target_quantile_grid_size", None))
        return cls(
            target_dist=target_dist,
            n_grid=n_grid,
            quantile_eps=quantile_eps,
            target_quantile_mode=mode,
            target_quantile_grid_size=grid_size,
        )

    def _cache_key(self, instance):
        return (
            instance.name,
            instance.K,
            self.n_grid,
            self.quantile_eps,
            self.target_quantile_mode,
            self.target_quantile_grid_size,
        )

    @staticmethod
    def _ensure_valid_cdf(cdf_vals):
        cdf_vals = np.asarray(cdf_vals, dtype=float)
        cdf_vals = np.maximum.accumulate(cdf_vals)
        cdf_vals = np.clip(cdf_vals, 0.0, 1.0)
        cdf_vals[0] = 0.0
        cdf_vals[-1] = 1.0
        return cdf_vals

    @staticmethod
    def _cdf_to_quantiles(x_grid, cdf_vals, u_grid):
        cdf_vals = WassersteinUtility._ensure_valid_cdf(cdf_vals)
        unique_cdf, idx = np.unique(cdf_vals, return_index=True)
        unique_x = x_grid[idx]
        if unique_cdf[0] > 0.0:
            unique_cdf = np.insert(unique_cdf, 0, 0.0)
            unique_x = np.insert(unique_x, 0, x_grid[0])
        if unique_cdf[-1] < 1.0:
            unique_cdf = np.append(unique_cdf, 1.0)
            unique_x = np.append(unique_x, x_grid[-1])
        return np.interp(u_grid, unique_cdf, unique_x)

    def _build_target_ppf(self, x_grid, u_grid):
        mode = self.target_quantile_mode
        if mode in {"auto", "formula"}:
            try:
                ppf_vals = np.asarray(self.target_dist.ppf(u_grid), dtype=float)
                if np.all(np.isfinite(ppf_vals)):
                    return ppf_vals
                if mode == "formula":
                    raise ValueError("Target ppf returned non-finite values in formula mode.")
            except Exception:
                if mode == "formula":
                    raise
        lo, hi = self.target_dist.support_bounds(self.quantile_eps)
        xq = np.linspace(lo, hi, self.target_quantile_grid_size)
        cdfq = self._ensure_valid_cdf(self.target_dist.cdf(xq))
        return self._cdf_to_quantiles(xq, cdfq, u_grid)

    def _prepare_cache(self, instance):
        key = self._cache_key(instance)
        if key in self._cache:
            return self._cache[key]

        eps = self.quantile_eps
        lows = []
        highs = []
        for arm in instance.arms:
            lo, hi = arm.support_bounds(eps)
            lows.append(lo)
            highs.append(hi)
        lo_t, hi_t = self.target_dist.support_bounds(eps)
        lows.append(lo_t)
        highs.append(hi_t)

        x_min = float(min(lows))
        x_max = float(max(highs))
        pad = 0.05 * max(1e-8, x_max - x_min)
        x_grid = np.linspace(x_min - pad, x_max + pad, self.n_grid)

        u_grid = np.linspace(eps, 1.0 - eps, self.n_grid)
        arm_cdfs = np.vstack([self._ensure_valid_cdf(arm.cdf(x_grid)) for arm in instance.arms])
        target_ppf = self._build_target_ppf(x_grid, u_grid)
        target_cdf = self._ensure_valid_cdf(self.target_dist.cdf(x_grid))
        dx = np.diff(x_grid)

        cache = {
            "x_grid": x_grid,
            "u_grid": u_grid,
            "dx": dx,
            "arm_cdfs": arm_cdfs,
            "target_ppf": target_ppf,
            "target_cdf": target_cdf,
        }
        self._cache[key] = cache
        return cache

    def exact_arm_cdf_grid(self, instance):
        return self._prepare_cache(instance)["arm_cdfs"]

    def target_cdf_grid(self, instance):
        return self._prepare_cache(instance)["target_cdf"]

    def grid_size(self, instance):
        return len(self._prepare_cache(instance)["x_grid"])

    def value_from_cdf_grid(self, instance, w, arm_cdf_grid):
        w = np.asarray(w, dtype=float)
        cache = self._prepare_cache(instance)
        x_grid = cache["x_grid"]
        u_grid = cache["u_grid"]
        target_ppf = cache["target_ppf"]

        mix_cdf = self._ensure_valid_cdf(np.dot(w, arm_cdf_grid))
        mix_ppf = self._cdf_to_quantiles(x_grid, mix_cdf, u_grid)
        w2sq = float(np.mean((mix_ppf - target_ppf) ** 2))
        return -w2sq

    def value(self, instance, w) -> float:
        cache = self._prepare_cache(instance)
        return self.value_from_cdf_grid(instance, w, cache["arm_cdfs"])

    @staticmethod
    def _stieltjes_expectation(function_values, cdf_values):
        cdf_values = WassersteinUtility._ensure_valid_cdf(cdf_values)
        jumps = np.empty_like(cdf_values)
        jumps[0] = cdf_values[0]
        jumps[1:] = cdf_values[1:] - cdf_values[:-1]
        return float(np.sum(np.asarray(function_values, dtype=float) * jumps))

    @staticmethod
    def _cumulative_trapezoid_zero(y, x):
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(y)
        if len(y) >= 2:
            out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
        return out

    def influence_function_from_cdf(self, instance, mix_cdf, *, center_mode: str = "mixture"):
        cache = self._prepare_cache(instance)
        x_grid = cache["x_grid"]
        target_ppf = cache["target_ppf"]
        mix_cdf = self._ensure_valid_cdf(mix_cdf)
        u = np.clip(mix_cdf, self.quantile_eps, 1.0 - self.quantile_eps)
        target_quantiles_on_mix = np.interp(u, cache["u_grid"], target_ppf)
        integrand = x_grid - target_quantiles_on_mix
        potential = self._cumulative_trapezoid_zero(integrand, x_grid)
        if_grid = -2.0 * potential

        if center_mode == "mixture":
            center = self._stieltjes_expectation(if_grid, mix_cdf)
        elif center_mode == "none":
            center = 0.0
        else:
            raise ValueError("center_mode must be 'mixture' or 'none'.")
        if_grid = if_grid - center

        return {
            "x_grid": x_grid,
            "mix_cdf": mix_cdf,
            "if_grid": if_grid,
            "transport_map": target_quantiles_on_mix,
            "integrand": integrand,
            "center": center,
        }

    def exact_influence_function(self, instance, w):
        cache = self._prepare_cache(instance)
        mix_cdf = self._ensure_valid_cdf(np.dot(np.asarray(w, dtype=float), cache["arm_cdfs"]))
        return self.influence_function_from_cdf(instance, mix_cdf, center_mode="mixture")

    def exact_gradient_coordinates(self, instance, w):
        cache = self._prepare_cache(instance)
        if_data = self.exact_influence_function(instance, w)
        if_grid = if_data["if_grid"]
        arm_cdfs = cache["arm_cdfs"]
        grad = np.array([
            self._stieltjes_expectation(if_grid, arm_cdfs[k]) for k in range(instance.K)
        ], dtype=float)
        grad -= np.dot(np.asarray(w, dtype=float), grad)
        return grad

    def evaluate_if_on_rewards(self, if_data, rewards):
        rewards = np.asarray(rewards, dtype=float)
        return np.interp(rewards, if_data["x_grid"], if_data["if_grid"], left=if_data["if_grid"][0], right=if_data["if_grid"][-1])
