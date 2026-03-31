import json
from pathlib import Path

import numpy as np


def run_single_experiment(instance, algorithm, T: int, seed: int = 123, utility=None):
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
    utility : optional
        Utility object with method utility.value(instance, w).

    Returns
    -------
    results : dict
        Dictionary containing actions, rewards, weight trajectory,
        and optionally utility trajectory.
    """
    if T <= 0:
        raise ValueError("T must be a positive integer.")

    rng = np.random.default_rng(seed)
    algorithm.reset(rng)

    actions = np.zeros(T, dtype=int)
    rewards = np.zeros(T, dtype=float)
    weights = np.zeros((T, instance.K), dtype=float)

    if utility is not None:
        utility_values = np.zeros(T, dtype=float)
    else:
        utility_values = None

    for t in range(T):
        action = algorithm.select_action()
        reward = instance.sample(action, rng)
        algorithm.update(action, reward)

        current_w = algorithm.current_weights()

        actions[t] = action
        rewards[t] = reward
        weights[t] = current_w

        if utility is not None:
            utility_values[t] = utility.value(instance, current_w)

    results = {
        "instance_name": instance.name,
        "algorithm_name": algorithm.name,
        "T": T,
        "seed": seed,
        "actions": actions,
        "rewards": rewards,
        "weights": weights,
    }

    if utility_values is not None:
        results["utility_values"] = utility_values

    return results


def save_results_npz(results: dict, filepath):
    """
    Save experiment results to a .npz file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "instance_name": results["instance_name"],
        "algorithm_name": results["algorithm_name"],
        "T": results["T"],
        "seed": results["seed"],
        "actions": results["actions"],
        "rewards": results["rewards"],
        "weights": results["weights"],
    }

    if "utility_values" in results:
        payload["utility_values"] = results["utility_values"]

    np.savez(filepath, **payload)


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
        "has_utility_values": "utility_values" in results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)