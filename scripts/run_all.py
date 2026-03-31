from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.arms import UniformArm
from src.instance import BanditInstance
from src.algorithms import SimpleMirrorAscent, UniformPolicy
from src.utilities import VarianceUtility
from src.offline_opt import solve_variance_optimum
from src.runner import run_single_experiment, save_results_npz, save_metadata_json
from src.plots import save_figure


def build_instance():
    arms = [
        UniformArm(0.0, 1.0),
        UniformArm(0.2, 0.8),
        UniformArm(-0.5, 1.0),
    ]
    return BanditInstance(arms, name="toy_uniform_instance")


def build_algorithms(K: int):
    return {
        "simple_mirror_ascent": SimpleMirrorAscent(K=K, eta=0.2),
        "uniform": UniformPolicy(K=K),
    }


def plot_mean_weight_trajectories_by_algorithm(weight_dict, instance_name):
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

        ax.set_xlabel("t")
        ax.set_ylabel("mean weight")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{algorithm_name} on {instance_name}")
        ax.legend()

    fig.tight_layout()
    return fig, axes


def plot_mean_utility_by_algorithm(utility_dict, u_star, instance_name):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for algorithm_name, utility_list in utility_dict.items():
        utilities = np.stack(utility_list, axis=0)
        mean_utilities = utilities.mean(axis=0)
        ax.plot(mean_utilities, label=algorithm_name)

    ax.axhline(u_star, linestyle="--", label="optimal utility")
    ax.set_xlabel("t")
    ax.set_ylabel("mean utility")
    ax.set_title(f"Mean variance utility on {instance_name}")
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_mean_utility_gap_by_algorithm(gap_dict, instance_name):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for algorithm_name, gap_list in gap_dict.items():
        gaps = np.stack(gap_list, axis=0)
        mean_gaps = gaps.mean(axis=0)
        ax.plot(mean_gaps, label=algorithm_name)

    ax.set_xlabel("t")
    ax.set_ylabel("mean utility gap")
    ax.set_title(f"Mean utility gap on {instance_name}")
    ax.legend()
    fig.tight_layout()

    return fig, ax


def main():
    T = 500
    seeds = [101, 202, 303, 404, 505]

    instance = build_instance()
    utility = VarianceUtility()
    w_star, u_star, opt_result = solve_variance_optimum(instance, utility)
    algorithms = build_algorithms(instance.K)

    raw_dir = ROOT / "results" / "raw" / "run_all"
    processed_dir = ROOT / "results" / "processed"
    fig_dir = ROOT / "results" / "figures"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    weight_dict = {name: [] for name in algorithms}
    utility_dict = {name: [] for name in algorithms}
    gap_dict = {name: [] for name in algorithms}

    for algorithm_name in algorithms:
        algo_raw_dir = raw_dir / algorithm_name
        algo_raw_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            algorithm = build_algorithms(instance.K)[algorithm_name]

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

            final_utility = float(utility_values[-1])
            mean_utility = float(np.mean(utility_values))
            final_gap = float(utility_gap_traj[-1])
            mean_gap = float(np.mean(utility_gap_traj))
            l1_error = float(np.sum(np.abs(final_weights - w_star)))

            summary_rows.append(
                {
                    "seed": seed,
                    "instance_name": results["instance_name"],
                    "algorithm_name": results["algorithm_name"],
                    "T": results["T"],
                    "u_star": float(u_star),
                    "final_utility": final_utility,
                    "mean_utility": mean_utility,
                    "final_gap": final_gap,
                    "mean_gap": mean_gap,
                    "l1_error_to_w_star": l1_error,
                    **{
                        f"final_w_{k}": float(final_weights[k])
                        for k in range(instance.K)
                    },
                    **{
                        f"w_star_{k}": float(w_star[k])
                        for k in range(instance.K)
                    },
                }
            )

            weight_dict[algorithm_name].append(results["weights"])
            utility_dict[algorithm_name].append(utility_values)
            gap_dict[algorithm_name].append(utility_gap_traj)

            print(
                f"Finished algorithm={algorithm_name}, seed={seed} | "
                f"final_gap={final_gap:.6f} | "
                f"l1_error={l1_error:.6f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = processed_dir / "run_all_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    batch_metadata = {
        "instance_name": instance.name,
        "utility_name": "variance",
        "algorithm_names": list(algorithms.keys()),
        "T": T,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "w_star": w_star.tolist(),
        "u_star": float(u_star),
        "optimizer_success": bool(opt_result.success),
        "optimizer_message": str(opt_result.message),
    }
    with open(processed_dir / "run_all_metadata.json", "w", encoding="utf-8") as f:
        json.dump(batch_metadata, f, indent=2)

    fig1, axes1 = plot_mean_weight_trajectories_by_algorithm(
        weight_dict=weight_dict,
        instance_name=instance.name,
    )
    save_figure(fig1, fig_dir / "run_all_mean_weights_by_algorithm.png")
    plt.close(fig1)

    fig2, ax2 = plot_mean_utility_by_algorithm(
        utility_dict=utility_dict,
        u_star=u_star,
        instance_name=instance.name,
    )
    save_figure(fig2, fig_dir / "run_all_mean_utility.png")
    plt.close(fig2)

    fig3, ax3 = plot_mean_utility_gap_by_algorithm(
        gap_dict=gap_dict,
        instance_name=instance.name,
    )
    save_figure(fig3, fig_dir / "run_all_mean_utility_gap.png")
    plt.close(fig3)

    print("\nBatch experiment completed.")
    print(f"Optimal weights w*: {np.round(w_star, 6)}")
    print(f"Optimal utility U(w*): {u_star:.6f}")
    print(f"Raw per-seed results saved to: {raw_dir}")
    print(f"Summary CSV saved to: {summary_path}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()