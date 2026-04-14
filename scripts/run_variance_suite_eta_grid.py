from pathlib import Path
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import load_yaml_config, should_save_raw_output
from src.algorithms import (
    VarianceMirrorAscent,
    VarianceIFAscent,
    WassersteinMirrorAscent,
    WassersteinIFAscent,
)
from src.utilities import VarianceUtility, WassersteinUtility
from src.offline_opt import solve_utility_optimum_gamma
from src.instance_factory import build_bandit_instance
from src.runner import run_single_experiment, save_results_npz, save_metadata_json
from src.plots import save_figure, plot_avg_weight_gap_by_eta


def build_utility_from_config(config):
    utility_name = config["utility"]["name"]
    if utility_name == "variance":
        return VarianceUtility()
    if utility_name == "wasserstein":
        return WassersteinUtility.from_config(config["utility"])
    raise ValueError(f"Unsupported utility.name: {utility_name}")


def build_algorithm_from_config(algorithm_cfg, K: int, gamma: float, eta0_override=None):
    name = algorithm_cfg["name"]
    eta0 = float(eta0_override if eta0_override is not None else algorithm_cfg.get("eta0", 0.2))

    if name == "variance_mirror_ascent":
        return VarianceMirrorAscent(K=K, eta0=eta0, gamma=gamma)
    if name == "variance_if_ascent":
        return VarianceIFAscent(
            K=K,
            eta0=eta0,
            gamma=gamma,
            prior_mean=float(algorithm_cfg.get("prior_mean", 0.0)),
            prior_second_moment=float(algorithm_cfg.get("prior_second_moment", 1.0)),
            prior_count=float(algorithm_cfg.get("prior_count", 1.0)),
        )
    if name == "wasserstein_mirror_ascent":
        return WassersteinMirrorAscent(K=K, eta0=eta0, gamma=gamma)
    if name == "wasserstein_if_ascent":
        return WassersteinIFAscent(
            K=K,
            eta0=eta0,
            gamma=gamma,
            prior_count=float(algorithm_cfg.get("prior_count", 1.0)),
            mixture_source=str(algorithm_cfg.get("mixture_source", "empirical")),
        )
    raise ValueError(f"Unknown algorithm name: {name}")


def resolve_seeds(config):
    if "seeds" in config and config["seeds"] is not None:
        return list(config["seeds"])
    n_seeds = int(config["n_seeds"])
    seed_start = int(config.get("seed_start", 101))
    seed_step = int(config.get("seed_step", 101))
    return [seed_start + i * seed_step for i in range(n_seeds)]


def resolve_eta_values(config, algorithm_cfgs):
    eta_values = config.get("eta_grid", None)
    if eta_values is not None:
        eta_values = [float(x) for x in eta_values]
        if len(eta_values) == 0:
            raise ValueError("eta_grid must be non-empty if provided.")
        return eta_values
    eta_candidates = [float(cfg["eta0"]) for cfg in algorithm_cfgs if "eta0" in cfg]
    return sorted(set(eta_candidates)) if eta_candidates else [0.2]


def algorithm_key_from_config(algorithm_cfg):
    name = algorithm_cfg["name"]
    if name == "wasserstein_if_ascent":
        mixture_source = str(algorithm_cfg.get("mixture_source", "empirical")).lower()
        if mixture_source == "exact":
            return "wasserstein_if_ascent_exact"
        if mixture_source == "empirical":
            return "wasserstein_if_ascent_empirical"
        raise ValueError(f"Unsupported mixture_source: {mixture_source}")
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/variance_suite_eta_grid.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)

    experiment_name = config["experiment_name"]
    T = int(config["horizon"])
    seeds = resolve_seeds(config)
    gamma = float(config["constraint"]["gamma"])
    algorithm_cfgs = list(config["algorithms"])
    instance_specs = list(config["instances"])
    eta_values = resolve_eta_values(config, algorithm_cfgs)
    save_raw = should_save_raw_output(config)
    utility = build_utility_from_config(config)
    utility_name = config["utility"]["name"]

    raw_base = ROOT / config["output"]["raw_dir"] / experiment_name
    processed_base = ROOT / config["output"]["processed_dir"] / experiment_name
    figures_base = ROOT / config["output"]["figures_dir"] / experiment_name

    if save_raw:
        raw_base.mkdir(parents=True, exist_ok=True)
    processed_base.mkdir(parents=True, exist_ok=True)
    figures_base.mkdir(parents=True, exist_ok=True)

    global_summary_rows = []

    for instance_spec in instance_specs:
        instance = build_bandit_instance(instance_spec)
        instance_name = instance.name
        w_star, u_star, opt_result = solve_utility_optimum_gamma(
            instance=instance,
            utility=utility,
            gamma=gamma,
        )

        summary_rows = []
        algorithm_keys = [algorithm_key_from_config(cfg) for cfg in algorithm_cfgs]
        avg_weight_gap_by_eta = {
            eta: {k: [] for k in algorithm_keys}
            for eta in eta_values
        }

        instance_raw_dir = None
        if save_raw:
            instance_raw_dir = raw_base / instance_name
            instance_raw_dir.mkdir(parents=True, exist_ok=True)

        for eta in eta_values:
            for algorithm_cfg in algorithm_cfgs:
                algorithm_key = algorithm_key_from_config(algorithm_cfg)

                algo_raw_dir = None
                if save_raw:
                    algo_raw_dir = instance_raw_dir / f"eta_{eta:g}" / algorithm_key
                    algo_raw_dir.mkdir(parents=True, exist_ok=True)

                for seed in seeds:
                    algorithm = build_algorithm_from_config(
                        algorithm_cfg,
                        K=instance.K,
                        gamma=gamma,
                        eta0_override=eta,
                    )
                    results = run_single_experiment(
                        instance=instance,
                        algorithm=algorithm,
                        T=T,
                        seed=seed,
                        utility=utility,
                    )

                    algorithm_name = results["algorithm_name"]

                    final_weights = results["weights"][-1]
                    final_avg_weights = results["average_weights"][-1]
                    utility_values = results["utility_values"]
                    avg_weight_utility_values = results["avg_weight_utility_values"]

                    utility_gap_traj = u_star - utility_values
                    avg_weight_gap_traj = u_star - avg_weight_utility_values
                    cumulative_regret_traj = np.cumsum(utility_gap_traj)
                    t_grid = np.arange(1, len(cumulative_regret_traj) + 1)
                    time_avg_gap_traj = cumulative_regret_traj / t_grid

                    results["utility_gap_values"] = utility_gap_traj
                    results["avg_weight_gap_values"] = avg_weight_gap_traj
                    results["cumulative_regret_values"] = cumulative_regret_traj
                    results["time_avg_gap_values"] = time_avg_gap_traj
                    results["final_average_weights"] = final_avg_weights.copy()

                    if save_raw:
                        save_results_npz(results, algo_raw_dir / f"seed_{seed}.npz")
                        save_metadata_json(results, algo_raw_dir / f"seed_{seed}.json")

                    row = {
                        "experiment_name": experiment_name,
                        "instance_name": instance.name,
                        "algorithm_name": algorithm_name,
                        "eta0": float(eta),
                        "seed": seed,
                        "T": results["T"],
                        "gamma": gamma,
                        "u_star": float(u_star),
                        "final_utility": float(utility_values[-1]),
                        "mean_utility": float(np.mean(utility_values)),
                        "final_avg_utility": float(avg_weight_utility_values[-1]),
                        "final_gap": float(utility_gap_traj[-1]),
                        "mean_gap": float(np.mean(utility_gap_traj)),
                        "final_avg_weight_gap": float(avg_weight_gap_traj[-1]),
                        "final_time_avg_gap": float(time_avg_gap_traj[-1]),
                        "final_cumulative_regret": float(cumulative_regret_traj[-1]),
                        "l1_error_to_w_star": float(np.sum(np.abs(final_avg_weights - w_star))),
                        "final_bias_inf": float(results["bias_inf_values"][-1]) if "bias_inf_values" in results else np.nan,
                        "final_bias_l2": float(results["bias_l2_values"][-1]) if "bias_l2_values" in results else np.nan,
                        **{f"final_w_{k}": float(final_weights[k]) for k in range(instance.K)},
                        **{f"final_avg_w_{k}": float(final_avg_weights[k]) for k in range(instance.K)},
                        **{f"w_star_{k}": float(w_star[k]) for k in range(instance.K)},
                    }

                    summary_rows.append(row)
                    global_summary_rows.append(row)
                    avg_weight_gap_by_eta[eta][algorithm_name].append(avg_weight_gap_traj)

                    print(
                        f"Finished instance={instance_name}, eta0={eta:g}, algorithm={algorithm_name}, seed={seed} "
                        f"| final_gap={row['final_gap']:.6f} | final_regret={row['final_cumulative_regret']:.6f}"
                    )

        pd.DataFrame(summary_rows).to_csv(processed_base / f"{instance_name}_summary.csv", index=False)

        with open(processed_base / f"{instance_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_name": experiment_name,
                    "instance_name": instance.name,
                    "utility_name": utility_name,
                    "algorithm_names": algorithm_keys,
                    "eta_values": eta_values,
                    "T": T,
                    "seeds": seeds,
                    "gamma": gamma,
                    "w_star": w_star.tolist(),
                    "u_star": float(u_star),
                    "optimizer_success": bool(opt_result.success),
                    "optimizer_message": str(opt_result.message),
                    "save_raw": save_raw,
                },
                f,
                indent=2,
            )

        fig, _ = plot_avg_weight_gap_by_eta(
            avg_weight_gap_by_eta=avg_weight_gap_by_eta,
            eta_values=eta_values,
            instance_name=instance.name,
            show=False,
        )
        save_figure(fig, figures_base / f"{instance_name}_avg_weight_gap_by_eta.pdf")
        plt.close(fig)

    pd.DataFrame(global_summary_rows).to_csv(processed_base / "all_instances_summary.csv", index=False)


if __name__ == "__main__":
    main()