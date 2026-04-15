import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "nature"])
# If LaTeX is not available from PATH, uncomment the next line:
# plt.rcParams["text.usetex"] = False

from matplotlib.lines import Line2D
from pathlib import Path


ALGORITHM_LABELS = {
    "variance_mirror_ascent": "Exact IF ascent",
    "variance_if_ascent": "Estimated IF ascent",
    "wasserstein_mirror_ascent": "Mirror ascent",
    "wasserstein_if_ascent_exact": "Exact IF ascent",
    "wasserstein_if_ascent_empirical": "Estimated IF ascent",
}


def pretty_algorithm_name(name: str) -> str:
    return ALGORITHM_LABELS.get(name, name)


def _mean_and_standard_error(trajectories):
    x = np.asarray(trajectories)
    if x.ndim != 2:
        raise ValueError("trajectories must have shape (n_runs, T).")

    n_runs = x.shape[0]
    mean_x = np.mean(x, axis=0)

    if n_runs == 1:
        return mean_x, mean_x, mean_x

    std_x = np.std(x, axis=0, ddof=1)
    se_x = std_x / np.sqrt(n_runs)

    lower = mean_x - se_x
    upper = mean_x + se_x
    return mean_x, lower, upper


def plot_mean_with_se(ax, trajectories, label: str, alpha: float = 0.20):
    mean_x, lower, upper = _mean_and_standard_error(trajectories)
    t = np.arange(len(mean_x))
    ax.plot(t, mean_x, label=label)
    ax.fill_between(t, lower, upper, alpha=alpha)


def plot_mean_weight_trajectories_by_algorithm(
    weight_dict,
    instance_name,
    gamma,
    show: bool = True,
):
    n_algorithms = len(weight_dict)

    fig, axes = plt.subplots(
        n_algorithms, 1, figsize=(4, 3 * n_algorithms), squeeze=False
    )

    for row, (algorithm_name, weight_list) in enumerate(weight_dict.items()):
        ax = axes[row, 0]

        if len(weight_list) == 0:
            ax.set_visible(False)
            continue

        weights = np.stack(weight_list, axis=0)
        mean_weights = weights.mean(axis=0)
        _, K = mean_weights.shape

        for k in range(K):
            ax.plot(mean_weights[:, k])

        ax.axhline(
            gamma,
            linestyle="--",
            linewidth=1.5,
            color="black",
        )

        algo_label = pretty_algorithm_name(algorithm_name)
        ax.set_xlabel("Step (t)")
        ax.set_ylabel("Mean Weight")
        ax.set_title(f"{algo_label} on {instance_name}")

        custom_handles = [
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="weights"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label=r"$\gamma$"),
        ]
        ax.legend(handles=custom_handles)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_mean_utility_by_algorithm(utility_dict, u_star, instance_name, show: bool = True):
    fig, ax = plt.subplots(figsize=(4, 3))

    for algorithm_name, utility_list in utility_dict.items():
        if len(utility_list) == 0:
            continue
        utilities = np.stack(utility_list, axis=0)
        algo_label = pretty_algorithm_name(algorithm_name)
        plot_mean_with_se(ax, utilities, label=algo_label)

    ax.axhline(u_star, linestyle="--", label="Optimal Utility")
    ax.set_xlabel("Step (t)")
    ax.set_ylabel("Utility")
    ax.set_title(f"Utility on {instance_name}")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_mean_utility_gap_by_algorithm(gap_dict, instance_name, show: bool = True):
    fig, ax = plt.subplots(figsize=(4, 3))

    for algorithm_name, gap_list in gap_dict.items():
        if len(gap_list) == 0:
            continue
        gaps = np.stack(gap_list, axis=0)
        algo_label = pretty_algorithm_name(algorithm_name)
        plot_mean_with_se(ax, gaps, label=algo_label)

    ax.set_xlabel("Step (t)")
    ax.set_ylabel("Utility Gap")
    ax.set_title(f"Utility Gap on {instance_name}")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_mean_cumulative_regret_by_algorithm(regret_dict, instance_name, show: bool = True):
    fig, ax = plt.subplots(figsize=(4, 3))

    for algorithm_name, regret_list in regret_dict.items():
        if len(regret_list) == 0:
            continue
        regrets = np.stack(regret_list, axis=0)
        algo_label = pretty_algorithm_name(algorithm_name)
        plot_mean_with_se(ax, regrets, label=algo_label)

    ax.set_xlabel("Step (t)")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(f"Cumulative Regret on {instance_name}")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_mean_mc_bias_norm(
    mc_bias_dict,
    instance_name,
    ylabel=r"$\|B_t\|_\infty$",
    show: bool = True,
):
    fig, ax = plt.subplots(figsize=(4, 3))

    for algorithm_name, bias_list in mc_bias_dict.items():
        if len(bias_list) == 0:
            continue
        biases = np.stack(bias_list, axis=0)
        algo_label = pretty_algorithm_name(algorithm_name)
        plot_mean_with_se(ax, biases, label=algo_label)

    ax.set_xlabel("Step (t)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Monte Carlo Bias Norm on {instance_name}")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_avg_weight_gap_and_time_avg_gap_by_algorithm(
    avg_weight_gap_dict,
    time_avg_gap_dict,
    instance_name,
    show: bool = True,
):
    fig, ax = plt.subplots(figsize=(4, 3))

    for algorithm_name, gap_list in avg_weight_gap_dict.items():
        if len(gap_list) == 0:
            continue
        gaps = np.stack(gap_list, axis=0)
        algo_label = pretty_algorithm_name(algorithm_name)
        plot_mean_with_se(ax, gaps, label=f"{algo_label}: avg-iterate")

    for algorithm_name, gap_list in time_avg_gap_dict.items():
        if len(gap_list) == 0:
            continue
        gaps = np.stack(gap_list, axis=0)
        algo_label = pretty_algorithm_name(algorithm_name)
        plot_mean_with_se(ax, gaps, label=f"{algo_label}: time-avg")

    ax.set_xlabel("Step (t)")
    ax.set_ylabel("Gap")
    ax.set_title(f"Avg-Weight Gap and Time-Averaged Gap on {instance_name}")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def _diagnostic_x_grid(instance, target_dist, n_grid: int = 500, eps: float = 1e-4):
    lows = []
    highs = []
    for arm in instance.arms:
        lo, hi = arm.support_bounds(eps)
        lows.append(float(lo))
        highs.append(float(hi))

    lo_q, hi_q = target_dist.support_bounds(eps)
    lows.append(float(lo_q))
    highs.append(float(hi_q))

    x_min = min(lows)
    x_max = max(highs)
    pad = 0.05 * max(1e-8, x_max - x_min)
    return np.linspace(x_min - pad, x_max + pad, int(n_grid))


def exact_mixture_density(instance, weights, x_grid):
    weights = np.asarray(weights, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    density = np.zeros_like(x_grid, dtype=float)
    for wk, arm in zip(weights, instance.arms):
        density += float(wk) * np.asarray(arm.pdf(x_grid), dtype=float)
    return density


def weighted_empirical_mixture_histogram(
    actions,
    rewards,
    avg_weights,
    bins,
):
    actions = np.asarray(actions, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    avg_weights = np.asarray(avg_weights, dtype=float)

    K = len(avg_weights)
    counts = np.bincount(actions, minlength=K).astype(float)

    point_weights = np.zeros_like(rewards, dtype=float)
    for k in range(K):
        if counts[k] > 0:
            point_weights[actions == k] = avg_weights[k] / counts[k]

    mask = point_weights > 0
    data = rewards[mask]
    weights = point_weights[mask]

    if len(data) == 0:
        raise ValueError("No positive-weight empirical samples available for histogram.")

    weights = weights / np.sum(weights)

    hist, edges = np.histogram(
        data,
        bins=bins,
        weights=weights,
        density=True,
    )
    return hist, edges


def plot_wasserstein_distributional_diagnostic(
    instance,
    utility,
    w_star,
    avg_weights_T,
    actions,
    rewards,
    n_grid: int = 500,
    n_bins: int = 40,
    title: str | None = None,
    show: bool = True,
):
    x_grid = _diagnostic_x_grid(instance, utility.target_dist, n_grid=n_grid)

    exact_opt_density = exact_mixture_density(instance, w_star, x_grid)
    reference_density = np.asarray(utility.target_dist.pdf(x_grid), dtype=float)

    hist, edges = weighted_empirical_mixture_histogram(
        actions=actions,
        rewards=rewards,
        avg_weights=avg_weights_T,
        bins=n_bins,
    )

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    line_opt, = ax.plot(
        x_grid,
        exact_opt_density,
        linewidth=1.5,
        label=r"Exact optimum $P^{w_\gamma^\star}$",
        zorder=3,
    )
    ax.fill_between(
        x_grid,
        0.0,
        exact_opt_density,
        alpha=0.20,
        color=line_opt.get_color(),
        zorder=1,
    )

    line_q, = ax.plot(
        x_grid,
        reference_density,
        linewidth=1.5,
        label=r"Reference $Q$",
        zorder=3,
    )
    ax.fill_between(
        x_grid,
        0.0,
        reference_density,
        alpha=0.20,
        color=line_q.get_color(),
        zorder=1,
    )

    ax.stairs(
        hist,
        edges,
        linewidth=1.8,
        label=r"Empirical $\widehat P^{\bar w_T}$",
        zorder=4,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    if title is None:
        title = f"Distributional diagnostic on {instance.name}"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def save_figure(fig, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight", format="pdf")