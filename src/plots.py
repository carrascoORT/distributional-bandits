import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','nature'])
from pathlib import Path


def plot_weight_trajectories(results: dict, title: str = None, show: bool = True):
    """
    Plot the weight trajectories stored in a results dictionary.
    """
    weights = np.asarray(results["weights"])
    T, K = weights.shape

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for k in range(K):
        ax.plot(weights[:, k], label=f"arm {k}")

    ax.set_xlabel("t")
    ax.set_ylabel("weight")
    ax.set_ylim(0.0, 1.0)

    if title is None:
        title = f"Weights - {results['algorithm_name']} on {results['instance_name']}"
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_cumulative_rewards(results: dict, title: str = None, show: bool = True):
    """
    Plot cumulative rewards from a results dictionary.
    """
    rewards = np.asarray(results["rewards"])
    cumulative_rewards = np.cumsum(rewards)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cumulative_rewards)

    ax.set_xlabel("t")
    ax.set_ylabel("cumulative reward")

    if title is None:
        title = f"Cumulative rewards - {results['algorithm_name']}"
    ax.set_title(title)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_utility_trajectory(results: dict, title: str = None, show: bool = True):
    """
    Plot utility values from a results dictionary.
    """
    if "utility_values" not in results:
        raise ValueError("results does not contain 'utility_values'.")

    utility_values = np.asarray(results["utility_values"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(utility_values)

    ax.set_xlabel("t")
    ax.set_ylabel("utility")

    if title is None:
        title = f"Utility - {results['algorithm_name']} on {results['instance_name']}"
    ax.set_title(title)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def _mean_and_standard_error(trajectories):
    """
    Compute the pointwise mean and standard error across runs.

    Parameters
    ----------
    trajectories : array-like, shape (n_runs, T)

    Returns
    -------
    mean_x : ndarray, shape (T,)
    lower : ndarray, shape (T,)
        mean - standard error
    upper : ndarray, shape (T,)
        mean + standard error
    """
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
    """
    Plot the mean trajectory with standard error bands on an existing axis.

    Parameters
    ----------
    ax : matplotlib axis
    trajectories : array-like, shape (n_runs, T)
    label : str
    alpha : float, default=0.20
        Transparency of the standard error band.
    """
    mean_x, lower, upper = _mean_and_standard_error(trajectories)
    t = np.arange(len(mean_x))
    ax.plot(t, mean_x, label=label)
    ax.fill_between(t, lower, upper, alpha=alpha)


def plot_mean_utility_by_algorithm(utility_dict, u_star, instance_name, show: bool = True):
    """
    Plot mean utility with standard error bands for each algorithm.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for algorithm_name, utility_list in utility_dict.items():
        utilities = np.stack(utility_list, axis=0)
        plot_mean_with_se(ax, utilities, label=algorithm_name)

    ax.axhline(u_star, linestyle="--", label="optimal utility")
    ax.set_xlabel("t")
    ax.set_ylabel("utility")
    ax.set_title(f"Variance utility on {instance_name}")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_mean_utility_gap_by_algorithm(gap_dict, instance_name, show: bool = True):
    """
    Plot mean utility gap with standard error bands for each algorithm.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for algorithm_name, gap_list in gap_dict.items():
        gaps = np.stack(gap_list, axis=0)
        plot_mean_with_se(ax, gaps, label=algorithm_name)

    ax.set_xlabel("t")
    ax.set_ylabel("utility gap")
    ax.set_title(f"Utility gap on {instance_name}")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_mean_cumulative_rewards_by_algorithm(reward_dict, instance_name, show: bool = True):
    """
    Plot mean cumulative rewards with standard error bands for each algorithm.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for algorithm_name, reward_list in reward_dict.items():
        rewards = np.stack(reward_list, axis=0)
        cum_rewards = np.cumsum(rewards, axis=1)
        plot_mean_with_se(ax, cum_rewards, label=algorithm_name)

    ax.set_xlabel("t")
    ax.set_ylabel("cumulative reward")
    ax.set_title(f"Cumulative rewards on {instance_name}")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_mean_weight_trajectories_by_algorithm(
    weight_dict,
    instance_name,
    gamma,
    show: bool = True,
):
    """
    Plot mean weight trajectories for each algorithm.
    No uncertainty bands here; each subplot shows the mean weight of each arm.
    A horizontal dashed line marks the threshold gamma.
    """
    n_algorithms = len(weight_dict)

    fig, axes = plt.subplots(
        n_algorithms, 1, figsize=(8, 4.5 * n_algorithms), squeeze=False
    )

    for row, (algorithm_name, weight_list) in enumerate(weight_dict.items()):
        ax = axes[row, 0]

        weights = np.stack(weight_list, axis=0)
        mean_weights = weights.mean(axis=0)

        T, K = mean_weights.shape

        for k in range(K):
            ax.plot(mean_weights[:, k], label=f"arm {k}")

        ax.axhline(
            gamma,
            linestyle="--",
            linewidth=1.5,
            color="black",
            label=r"$\gamma$",
        )

        ax.set_xlabel("t")
        ax.set_ylabel("mean weight")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{algorithm_name} on {instance_name}")

        handles, labels = ax.get_legend_handles_labels()
        gamma_idx = labels.index(r"$\gamma$")
        handles = handles[:gamma_idx] + handles[gamma_idx + 1 :] + [handles[gamma_idx]]
        labels = labels[:gamma_idx] + labels[gamma_idx + 1 :] + [labels[gamma_idx]]
        ax.legend(handles, labels)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


def load_results_npz(filepath):
    filepath = Path(filepath)

    with np.load(filepath, allow_pickle=True) as data:
        results = {
            "instance_name": str(data["instance_name"]),
            "algorithm_name": str(data["algorithm_name"]),
            "T": int(data["T"]),
            "seed": int(data["seed"]),
            "actions": data["actions"],
            "rewards": data["rewards"],
            "weights": data["weights"],
        }

        if "utility_values" in data.files:
            results["utility_values"] = data["utility_values"]
        if "average_weights" in data.files:
            results["average_weights"] = data["average_weights"]
        if "avg_weight_utility_values" in data.files:
            results["avg_weight_utility_values"] = data["avg_weight_utility_values"]

    return results


def plot_weights_from_file(filepath, title: str = None, show: bool = True):
    """
    Load a saved .npz file and plot weight trajectories.
    """
    results = load_results_npz(filepath)
    return plot_weight_trajectories(results, title=title, show=show)


def save_figure(fig, filepath):
    """
    Save a matplotlib figure to disk.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight", format='pdf')


def plot_avg_weight_gap_and_time_avg_gap_by_algorithm(
    avg_weight_gap_dict,
    time_avg_gap_dict,
    instance_name,
    show: bool = True,
):
    """
    Plot, for each algorithm:
      1) mean over seeds of the gap at the averaged iterate:
            U(w*) - U(bar w_t)
      2) mean over seeds of the time-averaged gap:
            (1/t) sum_{s=1}^t [U(w*) - U(w_s)]

    Both are shown with standard error bands.

    Parameters
    ----------
    avg_weight_gap_dict : dict
        algorithm_name -> list of arrays, each of shape (T,)
    time_avg_gap_dict : dict
        algorithm_name -> list of arrays, each of shape (T,)
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for algorithm_name, gap_list in avg_weight_gap_dict.items():
        gaps = np.stack(gap_list, axis=0)
        plot_mean_with_se(ax, gaps, label=f"{algorithm_name}: gap at averaged iterate")

    for algorithm_name, gap_list in time_avg_gap_dict.items():
        gaps = np.stack(gap_list, axis=0)
        plot_mean_with_se(ax, gaps, label=f"{algorithm_name}: time-averaged gap")

    ax.set_xlabel("t")
    ax.set_ylabel("gap")
    ax.set_title(f"Averaged-weight gap and time-averaged gap on {instance_name}")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax
