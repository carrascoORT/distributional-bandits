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
from src.plots import (
    save_figure,
    plot_mean_weight_trajectories_by_algorithm,
    plot_mean_utility_by_algorithm,
    plot_mean_utility_gap_by_algorithm,
    plot_avg_weight_gap_and_time_avg_gap_by_algorithm,
    plot_wasserstein_distributional_diagnostic,
)


def build_utility_from_config(config):
    utility_name = config["utility"]["name"]
    if utility_name == "variance":
        return VarianceUtility()
    if utility_name == "wasserstein":
        return WassersteinUtility.from_config(config["utility"])
    raise ValueError(f"Unsupported utility.name: {utility_name}")


def build_algorithm_from_config(algorithm_cfg, K: int, gamma: float):
    name = algorithm_cfg["name"]

    if name == "variance_mirror_ascent":
        return VarianceMirrorAscent(
            K=K,
            eta0=float(algorithm_cfg.get("eta0", 0.2)),
            gamma=gamma,
        )
    if name == "variance_if_ascent":
        return VarianceIFAscent(
            K=K,
            eta0=float(algorithm_cfg.get("eta0", 0.2)),
            gamma=gamma,
            prior_mean=float(algorithm_cfg.get("prior_mean", 0.0)),
            prior_second_moment=float(algorithm_cfg.get("prior_second_moment", 1.0)),
            prior_count=float(algorithm_cfg.get("prior_count", 1.0)),
        )
    if name == "wasserstein_mirror_ascent":
        return WassersteinMirrorAscent(
            K=K,
            eta0=float(algorithm_cfg.get("eta0", 0.2)),
            gamma=gamma,
        )
    if name == "wasserstein_if_ascent":
        return WassersteinIFAscent(
            K=K,
            eta0=float(algorithm_cfg.get("eta0", 0.2)),
            gamma=gamma,
            prior_count=float(algorithm_cfg.get("prior_count", 1.0)),
            mixture_source=str(algorithm_cfg.get("mixture_source", "empirical")),
        )
    raise ValueError(f"Unknown algorithm name: {name}")


def resolve_seeds(config):
    if "seeds" in config and config["seeds"] is not None:
        return list(config["seeds"])
    if "n_seeds" in config:
        n_seeds = int(config["n_seeds"])
        seed_start = int(config.get("seed_start", 101))
        seed_step = int(config.get("seed_step", 101))
        return [seed_start + i * seed_step for i in range(n_seeds)]
    raise ValueError("Config must contain either 'seeds' or 'n_seeds'.")


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
    parser.add_argument("--config", type=str, default="configs/variance_suite.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)

    experiment_name = config["experiment_name"]
    T = int(config["horizon"])
    seeds = resolve_seeds(config)
    gamma = float(config["constraint"]["gamma"])
    algorithm_cfgs = list(config["algorithms"])
    instance_specs = list(config["instances"])
    save_raw = should_save_raw_output(config)
    utility = build_utility_from_config(config)
    utility_name = config["utility"]["name"]

    diagnostics_cfg = config.get("diagnostics", {})
    density_cfg = diagnostics_cfg.get("density_plot", {})
    density_enabled = bool(density_cfg.get("enabled", False))
    density_instances = set(density_cfg.get("instances", []))
    density_seeds = set(density_cfg.get("seeds", []))
    density_algorithms = set(density_cfg.get("algorithms", []))
    density_bw_method = density_cfg.get("bw_method", None)
    density_n_grid = int(density_cfg.get("n_grid", 500))

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

        weight_dict = {k: [] for k in algorithm_keys}
        utility_dict = {k: [] for k in algorithm_keys}
        gap_dict = {k: [] for k in algorithm_keys}
        avg_weight_gap_dict = {k: [] for k in algorithm_keys}
        time_avg_gap_dict = {k: [] for k in algorithm_keys}

        instance_raw_dir = None
        if save_raw:
            instance_raw_dir = raw_base / instance_name
            instance_raw_dir.mkdir(parents=True, exist_ok=True)

        for algorithm_cfg in algorithm_cfgs:
            algorithm_key = algorithm_key_from_config(algorithm_cfg)

            algo_raw_dir = None
            if save_raw:
                algo_raw_dir = instance_raw_dir / algorithm_key
                algo_raw_dir.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                algorithm = build_algorithm_from_config(algorithm_cfg, K=instance.K, gamma=gamma)
                results = run_single_experiment(
                    instance=instance,
                    algorithm=algorithm,
                    T=T,
                    seed=seed,
                    utility=utility,
                )

                # Overwrite the ambiguous internal name with the resolved key.
                results["algorithm_name"] = algorithm_key

                if save_raw:
                    save_results_npz(results, algo_raw_dir / f"seed_{seed}.npz")
                    save_metadata_json(results, algo_raw_dir / f"seed_{seed}.json")

                final_weights = results["weights"][-1]
                final_avg_weights = results["average_weights"][-1]
                utility_values = results["utility_values"]
                utility_gap_traj = u_star - utility_values
                avg_weight_utility_values = results["avg_weight_utility_values"]
                avg_weight_gap_traj = u_star - avg_weight_utility_values
                cumulative_gap = np.cumsum(utility_gap_traj)
                t_grid = np.arange(1, len(cumulative_gap) + 1)
                time_avg_gap_traj = cumulative_gap / t_grid

                row = {
                    "experiment_name": experiment_name,
                    "instance_name": instance.name,
                    "algorithm_name": algorithm_key,
                    "seed": seed,
                    "T": results["T"],
                    "gamma": gamma,
                    "u_star": float(u_star),
                    "final_utility": float(utility_values[-1]),
                    "mean_utility": float(np.mean(utility_values)),
                    "final_gap": float(utility_gap_traj[-1]),
                    "mean_gap": float(np.mean(utility_gap_traj)),
                    "final_avg_weight_gap": float(avg_weight_gap_traj[-1]),
                    "final_time_avg_gap": float(time_avg_gap_traj[-1]),
                    "l1_error_to_w_star": float(np.sum(np.abs(final_avg_weights - w_star))),
                    **{f"final_w_{k}": float(final_weights[k]) for k in range(instance.K)},
                    **{f"final_avg_w_{k}": float(final_avg_weights[k]) for k in range(instance.K)},
                    **{f"w_star_{k}": float(w_star[k]) for k in range(instance.K)},
                }
                summary_rows.append(row)
                global_summary_rows.append(row)

                weight_dict[algorithm_key].append(results["weights"])
                utility_dict[algorithm_key].append(utility_values)
                gap_dict[algorithm_key].append(utility_gap_traj)
                avg_weight_gap_dict[algorithm_key].append(avg_weight_gap_traj)
                time_avg_gap_dict[algorithm_key].append(time_avg_gap_traj)

                # Distributional diagnostic: only exact optimum vs empirical KDE at bar w_T vs Q.
                should_make_density_plot = (
                    density_enabled
                    and utility_name == "wasserstein"
                    and algorithm_key == "wasserstein_if_ascent_empirical"
                    and (not density_instances or instance_name in density_instances)
                    and (not density_seeds or seed in density_seeds)
                    and (not density_algorithms or algorithm_key in density_algorithms)
                )

                if should_make_density_plot:
                    fig_diag, _ = plot_wasserstein_distributional_diagnostic(
                        instance=instance,
                        utility=utility,
                        w_star=w_star,
                        avg_weights_T=results["average_weights"][-1],
                        actions=results["actions"],
                        rewards=results["rewards"],
                        bw_method=density_bw_method,
                        n_grid=density_n_grid,
                        title=f"{instance_name} - seed {seed}",
                        show=False,
                    )
                    save_figure(
                        fig_diag,
                        figures_base / f"{instance_name}_{algorithm_key}_seed_{seed}_distributional_diagnostic.pdf",
                    )
                    plt.close(fig_diag)

                print(
                    f"Finished instance={instance_name}, algorithm={algorithm_key}, seed={seed} "
                    f"| final_gap={row['final_gap']:.6f}"
                )

        pd.DataFrame(summary_rows).to_csv(processed_base / f"{instance_name}_summary.csv", index=False)
        with open(processed_base / f"{instance_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_name": experiment_name,
                    "instance_name": instance.name,
                    "utility_name": utility_name,
                    "algorithm_names": algorithm_keys,
                    "T": T,
                    "seeds": seeds,
                    "n_seeds": len(seeds),
                    "gamma": gamma,
                    "w_star": w_star.tolist(),
                    "u_star": float(u_star),
                    "optimizer_success": bool(opt_result.success),
                    "optimizer_message": str(opt_result.message),
                    "save_raw": save_raw,
                    "density_plot_enabled": density_enabled,
                },
                f,
                indent=2,
            )

        fig1, _ = plot_mean_weight_trajectories_by_algorithm(
            weight_dict=weight_dict,
            instance_name=instance_name,
            gamma=gamma,
            show=False,
        )
        save_figure(fig1, figures_base / f"{instance_name}_mean_weights_by_algorithm.pdf")
        plt.close(fig1)

        fig2, _ = plot_mean_utility_by_algorithm(
            utility_dict=utility_dict,
            u_star=u_star,
            instance_name=instance.name,
            show=False,
        )
        save_figure(fig2, figures_base / f"{instance_name}_mean_utility_se_bands.pdf")
        plt.close(fig2)

        fig3, _ = plot_mean_utility_gap_by_algorithm(
            gap_dict=gap_dict,
            instance_name=instance.name,
            show=False,
        )
        save_figure(fig3, figures_base / f"{instance_name}_mean_utility_gap_se_bands.pdf")
        plt.close(fig3)

        fig4, _ = plot_avg_weight_gap_and_time_avg_gap_by_algorithm(
            avg_weight_gap_dict=avg_weight_gap_dict,
            time_avg_gap_dict=time_avg_gap_dict,
            instance_name=instance.name,
            show=False,
        )
        save_figure(fig4, figures_base / f"{instance_name}_avg_weight_and_time_avg_gap.pdf")
        plt.close(fig4)

    pd.DataFrame(global_summary_rows).to_csv(processed_base / "all_instances_summary.csv", index=False)


if __name__ == "__main__":
    main()