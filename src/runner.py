import json
from pathlib import Path

import numpy as np


def run_single_experiment(instance, algorithm, T: int, seed: int = 123, utility=None):
    """
    Run one trajectory of a bandit algorithm on one bandit instance.

    Returns actions, rewards, weights, and optionally utility-based summaries.
    """
    if T <= 0:
        raise ValueError("T must be a positive integer.")

    rng = np.random.default_rng(seed)

    try:
        algorithm.reset(rng, instance=instance)
    except TypeError:
        algorithm.reset(rng)

    actions = np.zeros(T, dtype=int)
    rewards = np.zeros(T, dtype=float)
    weights = np.zeros((T, instance.K), dtype=float)

    if utility is not None:
        utility_values = np.zeros(T, dtype=float)
        average_weights = np.zeros((T, instance.K), dtype=float)
        avg_weight_utility_values = np.zeros(T, dtype=float)
    else:
        utility_values = None
        average_weights = None
        avg_weight_utility_values = None

    running_weight_sum = np.zeros(instance.K, dtype=float)

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

            running_weight_sum += current_w
            avg_w_t = running_weight_sum / (t + 1)

            average_weights[t] = avg_w_t
            avg_weight_utility_values[t] = utility.value(instance, avg_w_t)

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
        results["average_weights"] = average_weights
        results["avg_weight_utility_values"] = avg_weight_utility_values

    return results


def save_results_npz(results: dict, filepath):
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
    if "average_weights" in results:
        payload["average_weights"] = results["average_weights"]
    if "avg_weight_utility_values" in results:
        payload["avg_weight_utility_values"] = results["avg_weight_utility_values"]

    np.savez(filepath, **payload)


def save_metadata_json(results: dict, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "instance_name": results["instance_name"],
        "algorithm_name": results["algorithm_name"],
        "T": int(results["T"]),
        "seed": int(results["seed"]),
        "K": int(results["weights"].shape[1]),
        "has_utility_values": "utility_values" in results,
        "has_average_weights": "average_weights" in results,
        "has_avg_weight_utility_values": "avg_weight_utility_values" in results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)