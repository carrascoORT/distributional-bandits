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
from src.runner import run_single_experiment, save_results_npz, save_metadata_json
from src.plots import save_figure


def build_instance():
    """
    Build a small toy instance.
    """
    arms = [
        UniformArm(0.0, 1.0),
        UniformArm(0.2, 0.8),
        UniformArm(-0.5, 1.0),
    ]
    return BanditInstance(arms, name="toy_uniform_instance")


def build_algorithms(K: int):
    """
    Return a dictionary of algorithms to compare.
    """
    return {
        "simple_mirror_ascent": SimpleMirrorAscent(K=K, eta=0.2),
        "uniform": UniformPolicy(K=K),
    }


def plot_mean_weight_trajectories_by_algorithm(weight_dict, instance_name):
    """
    Plot mean weight trajectories for each algorithm.

    Parameters
    ----------
    weight_dict : dict
        Maps algorithm_name -> list of weight arrays of shape (T, K).
    instance_name : str
        Name of the bandit instance.
    """
    n_algorithms = len(weight_dict)

    fig, axes = plt.subplots(
        n_algorithms, 1, figsize=(8, 4.5 * n_algorithms), squeeze=False
    )

    for row, (algorithm_name, weight_list) in enumerate(weight_dict.items()):
        ax = axes[row, 0]

        weights = np.stack(weight_list, axis=0)   # (n_seeds, T, K)
        mean_weights = weights.mean(axis=0)       # (T, K)

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


def plot_mean_cumulative_rewards_by_algorithm(reward_dict, instance_name):
    """
    Plot mean cumulative rewards for each algorithm.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for algorithm_name, reward_list in reward_dict.items():
        rewards = np.stack(reward_list, axis=0)          # (n_seeds, T)
        cum_rewards = np.cumsum(rewards, axis=1)         # (n_seeds, T)
        mean_cum_rewards = cum_rewards.mean(axis=0)      # (T,)

        ax.plot(mean_cum_rewards, label=algorithm_name)

    ax.set_xlabel("t")
    ax.set_ylabel("mean cumulative reward")
    ax.set_title(f"Mean cumulative rewards on {instance_name}")
    ax.legend()
    fig.tight_layout()

    return fig, ax


def main():
    # Parameters
    T = 500
    seeds = [101, 202, 303, 404, 505]

    instance = build_instance()
    algorithms = build_algorithms(instance.K)

    raw_dir = ROOT / "results" / "raw" / "run_all"
    processed_dir = ROOT / "results" / "processed"
    fig_dir = ROOT / "results" / "figures"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    weight_dict = {name: [] for name in algorithms}
    reward_dict = {name: [] for name in algorithms}

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
            )

            # Save one file per seed
            save_results_npz(results, algo_raw_dir / f"seed_{seed}.npz")
            save_metadata_json(results, algo_raw_dir / f"seed_{seed}.json")

            final_weights = results["weights"][-1]
            avg_reward = float(np.mean(results["rewards"]))
            cum_reward = float(np.sum(results["rewards"]))

            summary_rows.append(
                {
                    "seed": seed,
                    "instance_name": results["instance_name"],
                    "algorithm_name": results["algorithm_name"],
                    "T": results["T"],
                    "avg_reward": avg_reward,
                    "cum_reward": cum_reward,
                    **{
                        f"final_w_{k}": float(final_weights[k])
                        for k in range(instance.K)
                    },
                }
            )

            weight_dict[algorithm_name].append(results["weights"])
            reward_dict[algorithm_name].append(results["rewards"])

            print(
                f"Finished algorithm={algorithm_name}, seed={seed} | "
                f"avg_reward={avg_reward:.4f} | "
                f"final_weights={np.round(final_weights, 4)}"
            )

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_path = processed_dir / "run_all_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Save batch metadata
    batch_metadata = {
        "instance_name": instance.name,
        "algorithm_names": list(algorithms.keys()),
        "T": T,
        "seeds": seeds,
        "n_seeds": len(seeds),
    }
    with open(processed_dir / "run_all_metadata.json", "w", encoding="utf-8") as f:
        json.dump(batch_metadata, f, indent=2)

    # Save aggregated weight figure
    fig1, axes1 = plot_mean_weight_trajectories_by_algorithm(
        weight_dict=weight_dict,
        instance_name=instance.name,
    )
    save_figure(fig1, fig_dir / "run_all_mean_weights_by_algorithm.png")
    plt.close(fig1)

    # Save aggregated cumulative reward figure
    fig2, ax2 = plot_mean_cumulative_rewards_by_algorithm(
        reward_dict=reward_dict,
        instance_name=instance.name,
    )
    save_figure(fig2, fig_dir / "run_all_mean_cumulative_rewards.png")
    plt.close(fig2)

    print("\nBatch experiment completed.")
    print(f"Raw per-seed results saved to: {raw_dir}")
    print(f"Summary CSV saved to: {summary_path}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()