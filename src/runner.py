import json
from pathlib import Path

import numpy as np


def run_single_experiment(instance, algorithm, T: int, seed: int = 123):
    """
    Run one trajectory of a bandit algorithm on one bandit instance.

    Parameters
    ----------
    instance : BanditInstance
        The bandit environment.
    algorithm : BanditAlgorithm
        The bandit algorithm.
    T : int
        Time horizon.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Dictionary containing actions, rewards, and weight trajectory.
    """
    if T <= 0:
        raise ValueError("T must be a positive integer.")

    rng = np.random.default_rng(seed)
    algorithm.reset(rng)

    actions = np.zeros(T, dtype=int)
    rewards = np.zeros(T, dtype=float)
    weights = np.zeros((T, instance.K), dtype=float)

    for t in range(T):
        action = algorithm.select_action()
        reward = instance.sample(action, rng)
        algorithm.update(action, reward)

        actions[t] = action
        rewards[t] = reward
        weights[t] = algorithm.current_weights()

    results = {
        "instance_name": instance.name,
        "algorithm_name": algorithm.name,
        "T": T,
        "seed": seed,
        "actions": actions,
        "rewards": rewards,
        "weights": weights,
    }

    return results


def save_results_npz(results: dict, filepath):
    """
    Save experiment results to a .npz file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        filepath,
        instance_name=results["instance_name"],
        algorithm_name=results["algorithm_name"],
        T=results["T"],
        seed=results["seed"],
        actions=results["actions"],
        rewards=results["rewards"],
        weights=results["weights"],
    )


def save_metadata_json(results: dict, filepath):
    """
    Save only metadata to a small JSON file.
    Useful for quick inspection.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "instance_name": results["instance_name"],
        "algorithm_name": results["algorithm_name"],
        "T": int(results["T"]),
        "seed": int(results["seed"]),
        "K": int(results["weights"].shape[1]),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)