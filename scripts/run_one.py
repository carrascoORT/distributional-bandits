from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.arms import UniformArm
from src.instance import BanditInstance
from src.algorithms import SimpleMirrorAscent
from src.runner import run_single_experiment, save_results_npz, save_metadata_json
from src.plots import (
    plot_weight_trajectories,
    plot_cumulative_rewards,
    save_figure,
)


def main():
    # Build a small toy instance
    arms = [
        UniformArm(0.0, 1.0),
        UniformArm(0.2, 0.8),
        UniformArm(-0.5, 1.0),
    ]
    instance = BanditInstance(arms, name="toy_uniform_instance")

    # Build a simple algorithm
    algorithm = SimpleMirrorAscent(K=instance.K, eta=0.2)

    # Run one experiment
    results = run_single_experiment(
        instance=instance,
        algorithm=algorithm,
        T=200,
        seed=123,
    )

    # Save raw results
    raw_dir = ROOT / "results" / "raw"
    save_results_npz(results, raw_dir / "test_run.npz")
    save_metadata_json(results, raw_dir / "test_run.json")

    # Save figures
    fig_dir = ROOT / "results" / "figures"

    fig1, ax1 = plot_weight_trajectories(results, show=False)
    save_figure(fig1, fig_dir / "test_weights.png")
    plt.close(fig1)

    fig2, ax2 = plot_cumulative_rewards(results, show=False)
    save_figure(fig2, fig_dir / "test_cumulative_rewards.png")
    plt.close(fig2)

    print("Experiment completed.")
    print(f"Instance: {results['instance_name']}")
    print(f"Algorithm: {results['algorithm_name']}")
    print(f"Horizon: {results['T']}")
    print(f"Raw results saved to: {raw_dir}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()