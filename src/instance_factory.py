from src.arms import UniformArm, BetaArm, TriangularArm
from src.instance import BanditInstance


def build_variance_instance(name: str):
    """
    Return a named toy instance for variance-utility experiments.
    """

    if name == "toy_uniform_3":
        arms = [
            UniformArm(0.0, 1.0),
            UniformArm(0.2, 0.8),
            UniformArm(-0.5, 1.0),
        ]
        return BanditInstance(arms, name=name)

    elif name == "toy_uniform_5":
        arms = [
            UniformArm(0.0, 1.0),
            UniformArm(0.1, 0.9),
            UniformArm(0.2, 0.7),
            UniformArm(-0.5, 1.0),
            UniformArm(-0.2, 0.4),
        ]
        return BanditInstance(arms, name=name)

    elif name == "variance_boundary_4":
        arms = [
            UniformArm(0.0, 1.0),
            UniformArm(0.4, 0.6),
            UniformArm(-0.8, 1.2),
            UniformArm(0.2, 0.3),
        ]
        return BanditInstance(arms, name=name)

    elif name == "variance_sparse_5":
        arms = [
            UniformArm(0.0, 1.0),
            UniformArm(0.45, 0.55),
            UniformArm(-1.0, 1.0),
            UniformArm(0.1, 0.2),
            UniformArm(0.8, 0.9),
        ]
        return BanditInstance(arms, name=name)

    elif name == "beta_mixture_4":
        arms = [
            BetaArm(2.0, 8.0, a=0.0, b=1.0),
            BetaArm(8.0, 2.0, a=0.0, b=1.0),
            BetaArm(2.0, 2.0, a=0.0, b=1.0),
            BetaArm(20.0, 20.0, a=0.0, b=1.0),
        ]
        return BanditInstance(arms, name=name)

    elif name == "beta_shifted_5":
        arms = [
            BetaArm(2.0, 6.0, a=-1.0, b=1.0),
            BetaArm(6.0, 2.0, a=-1.0, b=1.0),
            BetaArm(3.0, 3.0, a=-1.0, b=1.0),
            BetaArm(12.0, 12.0, a=-1.0, b=1.0),
            BetaArm(1.5, 5.0, a=-1.0, b=1.0),
        ]
        return BanditInstance(arms, name=name)

    elif name == "triangular_4":
        arms = [
            TriangularArm(0.0, 0.2, 1.0),
            TriangularArm(0.0, 0.8, 1.0),
            TriangularArm(-1.0, 0.0, 1.0),
            TriangularArm(0.4, 0.5, 0.6),
        ]
        return BanditInstance(arms, name=name)

    elif name == "mixed_shapes_6":
        arms = [
            UniformArm(0.0, 1.0),
            BetaArm(2.0, 8.0, a=0.0, b=1.0),
            BetaArm(8.0, 2.0, a=0.0, b=1.0),
            TriangularArm(0.0, 0.2, 1.0),
            TriangularArm(0.0, 0.8, 1.0),
            UniformArm(0.45, 0.55),
        ]
        return BanditInstance(arms, name=name)

    else:
        raise ValueError(f"Unknown variance instance name: {name}")


def available_variance_instances():
    return [
        "toy_uniform_3",
        "toy_uniform_5",
        "variance_boundary_4",
        "variance_sparse_5",
        "beta_mixture_4",
        "beta_shifted_5",
        "triangular_4",
        "mixed_shapes_6",
    ]