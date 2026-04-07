from pathlib import Path
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import load_yaml_config
from src.algorithms import VarianceMirrorAscent, VarianceIFAscent
from src.utilities import VarianceUtility
from src.offline_opt import solve_variance_optimum_gamma
from src.instance_factory import build_variance_instance
from src.runner import run_single_experiment, save_results_npz, save_metadata_json
from src.plots import (
    save_figure,
    plot_mean_weight_trajectories_by_algorithm,
    plot_mean_utility_by_algorithm,
    plot_mean_utility_gap_by_algorithm,
    plot_avg_weight_gap_and_time_avg_gap_by_algorithm,
)


def build_algorithm_from_config(algorithm_cfg, K: int, gamma: float):
    name = algorithm_cfg["name"]

    if name == "variance_mirror_ascent":
        eta0 = float(algorithm_cfg.get("eta0", 0.2))
        return VarianceMirrorAscent(K=K, eta0=eta0, gamma=gamma)

    if name == "variance_if_ascent":
        eta0 = float(algorithm_cfg.get("eta0", 0.2))
        prior_mean = float(algorithm_cfg.get("prior_mean", 0.0))
        prior_second_moment = float(algorithm_cfg.get("prior_second_moment", 1.0))
        prior_count = float(algorithm_cfg.get("prior_count", 1.0))
        return VarianceIFAscent(
            K=K,
            eta0=eta0,
            gamma=gamma,
            prior_mean=prior_mean,
            prior_second_moment=prior_second_moment,
            prior_count=prior_count,
        )

    raise ValueError(f"Unknown algorithm name: {name}")


def resolve_seeds(config):
    if "seeds" in config and config["seeds"] is not None:
        return list(config["seeds"])

    if "n_seeds" in config:
        n_seeds = int(config["n_seeds"])
        if n_seeds <= 0:
            raise ValueError("n_seeds must be a positive integer.")

        seed_start = int(config.get("seed_start", 101))
        seed_step = int(config.get("seed_step", 101))
        return [seed_start + i * seed_step for i in range(n_seeds)]

    raise ValueError("Config must contain either 'seeds' or 'n_seeds'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/variance_suite.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    experiment_name = config["experiment_name"]
    T = int(config["horizon"])
    seeds = resolve_seeds(config)
    gamma = float(config["constraint"]["gamma"])
    utility_name = config["utility"]["name"]
    algorithm_cfgs = list(config["algorithms"])
    instance_names = list(config["instances"])

    if utility_name != "variance":
        raise ValueError("This script only supports utility.name = 'variance'.")

    raw_base = ROOT / config["output"]["raw_dir"] / experiment_name
    processed_base = ROOT / config["output"]["processed_dir"] / experiment_name
    figures_base = ROOT / config["output"]["figures_dir"] / experiment_name

    raw_base.mkdir(parents=True, exist_ok=True)
    processed_base.mkdir(parents=True, exist_ok=True)
    figures_base.mkdir(parents=True, exist_ok=True)

    utility = VarianceUtility()
    global_summary_rows = []

    for instance_name in instance_names:
        instance = build_variance_instance(instance_name)
        w_star, u_star, opt_result = solve_variance_optimum_gamma(
            instance=instance,
            utility=utility,
            gamma=gamma,
        )

        summary_rows = []

        weight_dict = {cfg["name"]: [] for cfg in algorithm_cfgs}
        utility_dict = {cfg["name"]: [] for cfg in algorithm_cfgs}
        gap_dict = {cfg["name"]: [] for cfg in algorithm_cfgs}
        avg_weight_gap_dict = {cfg["name"]: [] for cfg in algorithm_cfgs}
        time_avg_gap_dict = {cfg["name"]: [] for cfg in algorithm_cfgs}

        instance_raw_dir = raw_base / instance_name
        instance_raw_dir.mkdir(parents=True, exist_ok=True)

        for algorithm_cfg in algorithm_cfgs:
            algorithm_name = algorithm_cfg["name"]

            algo_raw_dir = instance_raw_dir / algorithm_name
            algo_raw_dir.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                algorithm = build_algorithm_from_config(
                    algorithm_cfg=algorithm_cfg,
                    K=instance.K,
                    gamma=gamma,
                )

                results = run_single_experiment(
                    instance=instance,
                    algorithm=algorithm,
                    T=T,
                    seed=seed,
                    utility=utility,
                )

                save_results_npz(results, algo_raw_dir / f"seed_{seed}.npz")
                save_metadata_json(results, algo_raw_dir / f"seed_{seed}.json")

                final_weights = results["weights"][-1]
                utility_values = results["utility_values"]
                utility_gap_traj = u_star - utility_values

                avg_weight_utility_values = results["avg_weight_utility_values"]
                avg_weight_gap_traj = u_star - avg_weight_utility_values

                cumulative_gap = np.cumsum(utility_gap_traj)
                t_grid = np.arange(1, len(cumulative_gap) + 1)
                time_avg_gap_traj = cumulative_gap / t_grid

                final_utility = float(utility_values[-1])
                mean_utility = float(np.mean(utility_values))
                final_gap = float(utility_gap_traj[-1])
                mean_gap = float(np.mean(utility_gap_traj))
                final_avg_weight_gap = float(avg_weight_gap_traj[-1])
                final_time_avg_gap = float(time_avg_gap_traj[-1])
                l1_error = float(np.sum(np.abs(final_weights - w_star)))

                row = {
                    "experiment_name": experiment_name,
                    "instance_name": instance.name,
                    "algorithm_name": results["algorithm_name"],
                    "seed": seed,
                    "T": results["T"],
                    "gamma": gamma,
                    "u_star": float(u_star),
                    "final_utility": final_utility,
                    "mean_utility": mean_utility,
                    "final_gap": final_gap,
                    "mean_gap": mean_gap,
                    "final_avg_weight_gap": final_avg_weight_gap,
                    "final_time_avg_gap": final_time_avg_gap,
                    "l1_error_to_w_star": l1_error,
                    **{f"final_w_{k}": float(final_weights[k]) for k in range(instance.K)},
                    **{f"w_star_{k}": float(w_star[k]) for k in range(instance.K)},
                }

                summary_rows.append(row)
                global_summary_rows.append(row)

                weight_dict[algorithm_name].append(results["weights"])
                utility_dict[algorithm_name].append(utility_values)
                gap_dict[algorithm_name].append(utility_gap_traj)
                avg_weight_gap_dict[algorithm_name].append(avg_weight_gap_traj)
                time_avg_gap_dict[algorithm_name].append(time_avg_gap_traj)

                print(
                    f"Finished instance={instance_name}, algorithm={algorithm_name}, seed={seed} | "
                    f"final_gap={final_gap:.6f} | "
                    f"final_avg_weight_gap={final_avg_weight_gap:.6f} | "
                    f"final_time_avg_gap={final_time_avg_gap:.6f} | "
                    f"l1_error={l1_error:.6f}"
                )

        instance_summary_df = pd.DataFrame(summary_rows)
        instance_summary_path = processed_base / f"{instance_name}_summary.csv"
        instance_summary_df.to_csv(instance_summary_path, index=False)

        instance_metadata = {
            "experiment_name": experiment_name,
            "instance_name": instance.name,
            "utility_name": utility_name,
            "algorithm_names": [cfg["name"] for cfg in algorithm_cfgs],
            "T": T,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "gamma": gamma,
            "w_star": w_star.tolist(),
            "u_star": float(u_star),
            "optimizer_success": bool(opt_result.success),
            "optimizer_message": str(opt_result.message),
            "uncertainty_bands": "standard_error",
            "config_path": args.config,
        }
        with open(processed_base / f"{instance_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(instance_metadata, f, indent=2)

        fig1, axes1 = plot_mean_weight_trajectories_by_algorithm(
            weight_dict=weight_dict,
            instance_name=instance_name,
            gamma=gamma,
            show=False,
        )
        save_figure(fig1, figures_base / f"{instance_name}_mean_weights_by_algorithm.pdf")
        plt.close(fig1)

        fig2, ax2 = plot_mean_utility_by_algorithm(
            utility_dict=utility_dict,
            u_star=u_star,
            instance_name=instance.name,
            show=False,
        )
        save_figure(fig2, figures_base / f"{instance_name}_mean_utility_se_bands.pdf")
        plt.close(fig2)

        fig3, ax3 = plot_mean_utility_gap_by_algorithm(
            gap_dict=gap_dict,
            instance_name=instance.name,
            show=False,
        )
        save_figure(fig3, figures_base / f"{instance_name}_mean_utility_gap_se_bands.pdf")
        plt.close(fig3)

        fig4, ax4 = plot_avg_weight_gap_and_time_avg_gap_by_algorithm(
            avg_weight_gap_dict=avg_weight_gap_dict,
            time_avg_gap_dict=time_avg_gap_dict,
            instance_name=instance.name,
            show=False,
        )
        save_figure(fig4, figures_base / f"{instance_name}_avg_weight_gap_and_time_avg_gap_se_bands.pdf")
        plt.close(fig4)

        print(f"\nCompleted instance: {instance_name}")
        print(f"  Optimal weights w*: {np.round(w_star, 6)}")
        print(f"  Optimal utility U(w*): {u_star:.6f}")
        print(f"  Summary saved to: {instance_summary_path}")

    global_summary_df = pd.DataFrame(global_summary_rows)
    global_summary_path = processed_base / "global_summary.csv"
    global_summary_df.to_csv(global_summary_path, index=False)

    global_metadata = {
        "experiment_name": experiment_name,
        "utility_name": utility_name,
        "instance_names": instance_names,
        "algorithm_names": [cfg["name"] for cfg in algorithm_cfgs],
        "T": T,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "gamma": gamma,
        "uncertainty_bands": "standard_error",
        "config_path": args.config,
    }
    with open(processed_base / "global_metadata.json", "w", encoding="utf-8") as f:
        json.dump(global_metadata, f, indent=2)

    print("\nVariance suite completed.")
    print(f"Raw results base: {raw_base}")
    print(f"Processed results base: {processed_base}")
    print(f"Figures base: {figures_base}")
    print(f"Global summary: {global_summary_path}")


if __name__ == "__main__":
    main()
