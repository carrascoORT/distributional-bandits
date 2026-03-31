from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.algorithms import VarianceMirrorAscent, UniformPolicy
from src.utilities import VarianceUtility
from src.offline_opt import solve_variance_optimum_gamma
from src.instance_factory import build_variance_instance
from src.runner import run_single_experiment, save_results_npz, save_metadata_json
from src.plots import (
    save_figure,
    plot_mean_weight_trajectories_by_algorithm,
    plot_mean_utility_by_algorithm,
    plot_mean_utility_gap_by_algorithm,
)


def build_algorithms(K: int, gamma: float):
    return {
        "variance_mirror_ascent": VarianceMirrorAscent(K=K, eta0=0.2, gamma=gamma),
        "uniform": UniformPolicy(K=K),
    }


def main():
    T = 500
    seeds = [101, 202, 303, 404, 505]
    gamma = 0.02
    instance_name = "variance_boundary_4"

    instance = build_variance_instance(instance_name)
    utility = VarianceUtility()
    w_star, u_star, opt_result = solve_variance_optimum_gamma(
        instance, utility, gamma=gamma
    )
    algorithms = build_algorithms(instance.K, gamma=gamma)

    raw_dir = ROOT / "results" / "raw" / instance_name
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
            algorithm = build_algorithms(instance.K, gamma=gamma)[algorithm_name]

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
                    "instance_name": instance.name,
                    "seed": seed,
                    "algorithm_name": results["algorithm_name"],
                    "T": results["T"],
                    "gamma": gamma,
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
                f"Finished instance={instance_name}, algorithm={algorithm_name}, seed={seed} | "
                f"final_gap={final_gap:.6f} | "
                f"l1_error={l1_error:.6f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = processed_dir / f"{instance_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    batch_metadata = {
        "instance_name": instance.name,
        "utility_name": "variance",
        "algorithm_names": list(algorithms.keys()),
        "T": T,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "gamma": gamma,
        "w_star": w_star.tolist(),
        "u_star": float(u_star),
        "optimizer_success": bool(opt_result.success),
        "optimizer_message": str(opt_result.message),
    }
    with open(processed_dir / f"{instance_name}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(batch_metadata, f, indent=2)

    fig1, axes1 = plot_mean_weight_trajectories_by_algorithm(
        weight_dict=weight_dict,
        instance_name=instance.name,
        show=False,
    )
    save_figure(fig1, fig_dir / f"{instance_name}_mean_weights_by_algorithm.png")
    plt.close(fig1)

    fig2, ax2 = plot_mean_utility_by_algorithm(
        utility_dict=utility_dict,
        u_star=u_star,
        instance_name=instance.name,
        show=False,
    )
    save_figure(fig2, fig_dir / f"{instance_name}_mean_utility.png")
    plt.close(fig2)

    fig3, ax3 = plot_mean_utility_gap_by_algorithm(
        gap_dict=gap_dict,
        instance_name=instance.name,
        show=False,
    )
    save_figure(fig3, fig_dir / f"{instance_name}_mean_utility_gap.png")
    plt.close(fig3)

    print("\nBatch experiment completed.")
    print(f"Instance: {instance_name}")
    print(f"Optimal weights w*: {np.round(w_star, 6)}")
    print(f"Optimal utility U(w*): {u_star:.6f}")
    print(f"Raw per-seed results saved to: {raw_dir}")
    print(f"Summary CSV saved to: {summary_path}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()