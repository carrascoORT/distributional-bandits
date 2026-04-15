import json
from pathlib import Path

import numpy as np


def run_single_experiment(
    instance,
    algorithm,
    T: int,
    seed: int = 123,
    utility=None,
    mc_bias_n: int | None = None,
):
    if T <= 0:
        raise ValueError("T must be a positive integer.")

    rng = np.random.default_rng(seed)
    mc_rng = np.random.default_rng(seed + 10_000_003)

    try:
        algorithm.reset(rng, instance=instance, utility=utility)
    except TypeError:
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

    mc_bias_vector_values = []
    mc_bias_inf_values = np.full(T, np.nan, dtype=float)
    mc_bias_l2_values = np.full(T, np.nan, dtype=float)

    for t in range(T):
        if (
            mc_bias_n is not None
            and mc_bias_n > 0
            and hasattr(algorithm, "estimate_conditional_bias_mc")
            and getattr(algorithm, "name", "") in {"variance_if_ascent", "wasserstein_if_ascent_empirical"}
        ):
            mc_diag = algorithm.estimate_conditional_bias_mc(mc_rng, n_mc=mc_bias_n)
            mc_bias_vector_values.append(np.asarray(mc_diag["mc_bias_vector"], dtype=float))
            mc_bias_inf_values[t] = float(mc_diag["mc_bias_inf"])
            mc_bias_l2_values[t] = float(mc_diag["mc_bias_l2"])

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
            avg_w_t = running_weight_sum / float(t + 1)
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

    if utility is not None:
        results["utility_values"] = utility_values
        results["average_weights"] = average_weights
        results["avg_weight_utility_values"] = avg_weight_utility_values

    if len(mc_bias_vector_values) > 0:
        results["mc_bias_vector_values"] = np.stack(mc_bias_vector_values, axis=0)
        results["mc_bias_inf_values"] = mc_bias_inf_values
        results["mc_bias_l2_values"] = mc_bias_l2_values

    return results


def save_results_npz(results: dict, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath, **results)


def save_metadata_json(results: dict, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        elif isinstance(value, (np.integer, np.floating)):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)