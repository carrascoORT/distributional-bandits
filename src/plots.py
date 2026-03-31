import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_weight_trajectories(results: dict, title: str = None, show: bool = True):
    """
    Plot the weight trajectories stored in a results dictionary.

    Parameters
    ----------
    results : dict
        Output of run_single_experiment.
    title : str, optional
        Plot title.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig, ax : matplotlib figure and axes
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


def load_results_npz(filepath):
    """
    Load a .npz results file and return a standard dictionary.
    """
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
    fig.savefig(filepath, bbox_inches="tight")