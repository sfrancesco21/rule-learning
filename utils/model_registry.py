from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict

from utils.fit_gmm import fit_gmm_model
from utils.fit_wm import fit_exp_decay_model, fit_last_m_model


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    fit_func: Callable[..., Dict[str, Any]]
    fit_defaults: Dict[str, Any]


MODEL_SPECS: Dict[str, ModelSpec] = {
    "gmm_no_forgetting_no_lr": ModelSpec(
        name="gmm_no_forgetting_no_lr",
        family="gmm",
        fit_func=partial(
            fit_gmm_model,
            config={
                "K0": 3,
                "K1": 3,
                "eps_cov": 1e-3,
                "use_forgetting": False,
                "use_lr_mu": False,
                "use_lr_cov": False,
                "tie_pull_lrs": False,
            },
        ),
        fit_defaults={"n_starts": 3, "num_steps": 4000, "lr": 5e-3},
    ),
    "gmm_no_forgetting_tied_lrs": ModelSpec(
        name="gmm_no_forgetting_tied_lrs",
        family="gmm",
        fit_func=partial(
            fit_gmm_model,
            config={
                "K0": 3,
                "K1": 3,
                "eps_cov": 1e-3,
                "use_forgetting": False,
                "use_lr_mu": True,
                "use_lr_cov": True,
                "tie_pull_lrs": True,
            },
        ),
        fit_defaults={"n_starts": 3, "num_steps": 4000, "lr": 5e-3},
    ),
    "gmm_forgetting_no_lr": ModelSpec(
        name="gmm_forgetting_no_lr",
        family="gmm",
        fit_func=partial(
            fit_gmm_model,
            config={
                "K0": 3,
                "K1": 3,
                "eps_cov": 1e-3,
                "use_forgetting": True,
                "use_lr_mu": False,
                "use_lr_cov": False,
                "tie_pull_lrs": False,
            },
        ),
        fit_defaults={"n_starts": 3, "num_steps": 4000, "lr": 5e-3},
    ),
    "gmm_forgetting_tied_lrs": ModelSpec(
        name="gmm_forgetting_tied_lrs",
        family="gmm",
        fit_func=partial(
            fit_gmm_model,
            config={
                "K0": 3,
                "K1": 3,
                "eps_cov": 1e-3,
                "use_forgetting": True,
                "use_lr_mu": True,
                "use_lr_cov": True,
                "tie_pull_lrs": True,
            },
        ),
        fit_defaults={"n_starts": 3, "num_steps": 4000, "lr": 5e-3},
    ),
    "wm_exponential_decay": ModelSpec(
        name="wm_exponential_decay",
        family="wm_exp",
        fit_func=partial(fit_exp_decay_model, use_dual_decay=False),
        fit_defaults={
            "decay_bounds": (1e-6, 10.0),
            "decay_time_bounds": (1e-6, 10.0),
            "decay_count_bounds": (1e-6, 10.0),
            "lapse_bounds": (1e-6, 0.49),
        },
    ),
    "wm_exponential_decay_dual": ModelSpec(
        name="wm_exponential_decay_dual",
        family="wm_exp",
        fit_func=partial(fit_exp_decay_model, use_dual_decay=True),
        fit_defaults={
            "decay_bounds": (1e-6, 10.0),
            "decay_time_bounds": (1e-6, 10.0),
            "decay_count_bounds": (1e-6, 10.0),
            "lapse_bounds": (1e-6, 0.49),
        },
    ),
    "wm_last_m": ModelSpec(
        name="wm_last_m",
        family="wm_last_m",
        fit_func=partial(fit_last_m_model, use_category_specific_memory=False),
        fit_defaults={
            "M_grid": range(1, 13),
            "M0_grid": range(1, 13),
            "M1_grid": range(1, 13),
            "lapse_bounds": (1e-6, 0.49),
        },
    ),
    "wm_last_m_category_specific": ModelSpec(
        name="wm_last_m_category_specific",
        family="wm_last_m",
        fit_func=partial(fit_last_m_model, use_category_specific_memory=True),
        fit_defaults={
            "M_grid": range(1, 13),
            "M0_grid": range(1, 13),
            "M1_grid": range(1, 13),
            "lapse_bounds": (1e-6, 0.49),
        },
    ),
}


def list_model_names() -> list[str]:
    return list(MODEL_SPECS.keys())


def get_model_spec(model_name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_name]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{model_name}'. Available models: {', '.join(list_model_names())}") from exc
