from src.arms import UniformArm, BetaArm, TriangularArm, GaussianArm
from src.instance import BanditInstance


PRESET_INSTANCE_NAMES = [
    "toy_uniform_3",
    "toy_uniform_5",
    "variance_boundary_4",
    "variance_sparse_5",
    "beta_mixture_4",
    "beta_shifted_5",
    "triangular_4",
    "mixed_shapes_6",
]


def _build_preset_variance_instance(name: str):
    if name == "toy_uniform_3":
        arms = [
            UniformArm(0.0, 1.0),
            UniformArm(0.2, 0.8),
            UniformArm(-0.5, 1.0),
        ]
        return BanditInstance(arms, name=name)

    if name == "toy_uniform_5":
        arms = [
            UniformArm(0.0, 1.0),
            UniformArm(0.1, 0.9),
            UniformArm(0.2, 0.7),
            UniformArm(-0.5, 1.0),
            UniformArm(-0.2, 0.4),
        ]
        return BanditInstance(arms, name=name)

    if name == "variance_boundary_4":
        arms = [
            UniformArm(0.0, 1.0),
            UniformArm(0.4, 0.6),
            UniformArm(-0.8, 1.2),
            UniformArm(0.2, 0.3),
        ]
        return BanditInstance(arms, name=name)

    if name == "variance_sparse_5":
        arms = [
            UniformArm(0.0, 1.0),
            UniformArm(0.45, 0.55),
            UniformArm(-1.0, 1.0),
            UniformArm(0.1, 0.2),
            UniformArm(0.8, 0.9),
        ]
        return BanditInstance(arms, name=name)

    if name == "beta_mixture_4":
        arms = [
            BetaArm(2.0, 8.0, a=0.0, b=1.0),
            BetaArm(8.0, 2.0, a=0.0, b=1.0),
            BetaArm(2.0, 2.0, a=0.0, b=1.0),
            BetaArm(20.0, 20.0, a=0.0, b=1.0),
        ]
        return BanditInstance(arms, name=name)

    if name == "beta_shifted_5":
        arms = [
            BetaArm(2.0, 6.0, a=-1.0, b=1.0),
            BetaArm(6.0, 2.0, a=-1.0, b=1.0),
            BetaArm(3.0, 3.0, a=-1.0, b=1.0),
            BetaArm(12.0, 12.0, a=-1.0, b=1.0),
            BetaArm(1.5, 5.0, a=-1.0, b=1.0),
        ]
        return BanditInstance(arms, name=name)

    if name == "triangular_4":
        arms = [
            TriangularArm(0.0, 0.2, 1.0),
            TriangularArm(0.0, 0.8, 1.0),
            TriangularArm(-1.0, 0.0, 1.0),
            TriangularArm(0.4, 0.5, 0.6),
        ]
        return BanditInstance(arms, name=name)

    if name == "mixed_shapes_6":
        arms = [
            UniformArm(0.0, 1.0),
            BetaArm(2.0, 8.0, a=0.0, b=1.0),
            BetaArm(8.0, 2.0, a=0.0, b=1.0),
            TriangularArm(0.0, 0.2, 1.0),
            TriangularArm(0.0, 0.8, 1.0),
            UniformArm(0.45, 0.55),
        ]
        return BanditInstance(arms, name=name)

    raise ValueError(f"Unknown variance instance name: {name}")


def build_distribution_from_spec(spec: dict):
    if not isinstance(spec, dict):
        raise TypeError("Distribution specification must be a dictionary.")

    family = spec.get("family", spec.get("dist", None))
    if family is None:
        raise ValueError("Distribution specification must define 'family' (or 'dist').")

    family = str(family).lower()

    if family == "uniform":
        return UniformArm(a=float(spec["a"]), b=float(spec["b"]))

    if family == "beta":
        if "loc" in spec or "scale" in spec:
            loc = float(spec.get("loc", 0.0))
            scale = float(spec.get("scale", 1.0))
            return BetaArm(
                alpha=float(spec["alpha"]),
                beta=float(spec["beta"]),
                a=loc,
                b=loc + scale,
            )
        return BetaArm(
            alpha=float(spec["alpha"]),
            beta=float(spec["beta"]),
            a=float(spec.get("a", 0.0)),
            b=float(spec.get("b", 1.0)),
        )

    if family == "triangular":
        return TriangularArm(
            left=float(spec["left"]),
            mode=float(spec["mode"]),
            right=float(spec["right"]),
        )

    if family == "gaussian":
        return GaussianArm(mu=float(spec["mu"]), sigma=float(spec["sigma"]))

    raise ValueError(f"Unknown arm family: {family}")


def _build_parametric_instance(instance_cfg: dict):
    name = str(instance_cfg.get("name", "custom_instance"))

    if "arms" in instance_cfg:
        arms = [build_distribution_from_spec(arm_cfg) for arm_cfg in instance_cfg["arms"]]
        return BanditInstance(arms, name=name)

    family = str(instance_cfg["family"]).lower()
    params = instance_cfg.get("params", {})

    if family == "beta":
        alpha_list = list(params["alpha"])
        beta_list = list(params["beta"])
        if len(alpha_list) != len(beta_list):
            raise ValueError(f"{name}: params.alpha and params.beta must have the same length.")

        if "loc" in params or "scale" in params:
            loc_list = list(params.get("loc", [0.0] * len(alpha_list)))
            scale_list = list(params.get("scale", [1.0] * len(alpha_list)))
            if not (len(loc_list) == len(scale_list) == len(alpha_list)):
                raise ValueError(f"{name}: beta loc/scale lists must match alpha length.")
            arms = [
                BetaArm(alpha=float(alpha), beta=float(beta), a=float(loc), b=float(loc) + float(scale))
                for alpha, beta, loc, scale in zip(alpha_list, beta_list, loc_list, scale_list)
            ]
            return BanditInstance(arms, name=name)

        a = float(params.get("a", 0.0))
        b = float(params.get("b", 1.0))
        arms = [
            BetaArm(alpha=float(alpha), beta=float(beta), a=a, b=b)
            for alpha, beta in zip(alpha_list, beta_list)
        ]
        return BanditInstance(arms, name=name)

    if family == "gaussian":
        mu_list = list(params["mu"])
        sigma_list = list(params["sigma"])
        if len(mu_list) != len(sigma_list):
            raise ValueError(f"{name}: params.mu and params.sigma must have the same length.")
        arms = [GaussianArm(mu=float(mu), sigma=float(sigma)) for mu, sigma in zip(mu_list, sigma_list)]
        return BanditInstance(arms, name=name)

    if family == "uniform":
        a_list = list(params["a"])
        b_list = list(params["b"])
        if len(a_list) != len(b_list):
            raise ValueError(f"{name}: params.a and params.b must have the same length.")
        arms = [UniformArm(a=float(a), b=float(b)) for a, b in zip(a_list, b_list)]
        return BanditInstance(arms, name=name)

    if family == "triangular":
        left_list = list(params["left"])
        mode_list = list(params["mode"])
        right_list = list(params["right"])
        if not (len(left_list) == len(mode_list) == len(right_list)):
            raise ValueError(f"{name}: params.left, params.mode, and params.right must have the same length.")
        arms = [
            TriangularArm(left=float(left), mode=float(mode), right=float(right))
            for left, mode, right in zip(left_list, mode_list, right_list)
        ]
        return BanditInstance(arms, name=name)

    raise ValueError(f"Unknown instance family: {family}")


def build_variance_instance(instance_spec):
    if isinstance(instance_spec, str):
        return _build_preset_variance_instance(instance_spec)
    if isinstance(instance_spec, dict):
        return _build_parametric_instance(instance_spec)
    raise TypeError("Instance specification must be either a string or a dictionary.")


def build_bandit_instance(instance_spec):
    return build_variance_instance(instance_spec)


def available_variance_instances():
    return list(PRESET_INSTANCE_NAMES)
