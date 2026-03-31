from pathlib import Path
import sys

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.arms import UniformArm
from src.instance import BanditInstance
from src.algorithms import SimpleMirrorAscent
from src.runner import run_single_experiment, save_results_npz, save_metadata_json


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

    # Save results
    output_dir = ROOT / "results" / "raw"
    save_results_npz(results, output_dir / "test_run.npz")
    save_metadata_json(results, output_dir / "test_run.json")

    print("Experiment completed.")
    print(f"Instance: {results['instance_name']}")
    print(f"Algorithm: {results['algorithm_name']}")
    print(f"Horizon: {results['T']}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()