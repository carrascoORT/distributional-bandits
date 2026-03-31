from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.arms import UniformArm
from src.instance import BanditInstance
from src.algorithms import SimpleMirrorAscent
from src.utilities import VarianceUtility
from src.offline_opt import solve_variance_optimum
from src.runner import run_single_experiment, save_results_npz, save_metadata_json
from src.plots import (
    plot_weight_trajectories,
    plot_cumulative_rewards,
    plot_utility_trajectory,
    save_figure,
)


def main():
    arms = [
        UniformArm(0.0, 1.0),
        UniformArm(0.2, 0.8),
        UniformArm(-0.5, 1.0),
    ]
    instance = BanditInstance(arms, name="toy_uniform_instance")
    utility = VarianceUtility()

    w_star, u_star, opt_result = solve_variance_optimum(instance, utility)

    algorithm = SimpleMirrorAscent(K=instance.K, eta=0.2)

    results = run_single_experiment(
        instance=instance,
        algorithm=algorithm,
        T=200,
        seed=123,
        utility=utility,
    )

    final_w = results["weights"][-1]
    final_utility = float(results["utility_values"][-1])
    utility_gap = float(u_star - final_utility)
    l1_error = float(np.sum(np.abs(final_w - w_star)))

    raw_dir = ROOT / "results" / "raw"
    save_results_npz(results, raw_dir / "test_run.npz")
    save_metadata_json(results, raw_dir / "test_run.json")

    fig_dir = ROOT / "results" / "figures"

    fig1, ax1 = plot_weight_trajectories(results, show=False)
    save_figure(fig1, fig_dir / "test_weights.png")
    plt.close(fig1)

    fig2, ax2 = plot_cumulative_rewards(results, show=False)
    save_figure(fig2, fig_dir / "test_cumulative_rewards.png")
    plt.close(fig2)

    fig3, ax3 = plot_utility_trajectory(results, show=False)
    save_figure(fig3, fig_dir / "test_utility.png")
    plt.close(fig3)

    print("Experiment completed.")
    print(f"Instance: {results['instance_name']}")
    print(f"Algorithm: {results['algorithm_name']}")
    print(f"Horizon: {results['T']}")
    print(f"Optimal weights w*: {np.round(w_star, 6)}")
    print(f"Optimal utility U(w*): {u_star:.6f}")
    print(f"Final utility: {final_utility:.6f}")
    print(f"Utility gap: {utility_gap:.6f}")
    print(f"L1 error to w*: {l1_error:.6f}")
    print(f"Raw results saved to: {raw_dir}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()