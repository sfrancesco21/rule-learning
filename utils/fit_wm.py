from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from utils.fit_common import aic_bic_from_ll, participant_accuracy, wm_expected_accuracy


def fit_exp_decay_model(
    *,
    participant_data: Dict[str, Any],
    participant_id: int,
    seed: int,
    use_dual_decay: bool = False,
    decay_bounds: tuple[float, float] = (1e-6, 10.0),
    decay_time_bounds: tuple[float, float] = (1e-6, 10.0),
    decay_count_bounds: tuple[float, float] = (1e-6, 10.0),
    lapse_bounds: tuple[float, float] = (1e-6, 0.49),
    **_: Any,
) -> Dict[str, Any]:
    from models.WM import ExponentialDecayInverseDistanceModel
    from utils.metrics import count_nonlost_trials
    from utils.mle_wm import fit_exp_decay_mle

    rng = np.random.default_rng(seed)
    if use_dual_decay:
        model = ExponentialDecayInverseDistanceModel(
            use_dual_decay=True,
            decay_time=0.1,
            decay_count=0.1,
        )
        x0 = {
            "decay_time": float(rng.uniform(*decay_time_bounds)),
            "decay_count": float(rng.uniform(*decay_count_bounds)),
            "lapse_rate": float(rng.uniform(*lapse_bounds)),
        }
    else:
        model = ExponentialDecayInverseDistanceModel(decay_rate=0.1)
        x0 = {
            "decay_rate": float(rng.uniform(*decay_bounds)),
            "lapse_rate": float(rng.uniform(*lapse_bounds)),
        }
    fit = fit_exp_decay_mle(
        model=model,
        data=participant_data,
        decay_bounds=decay_bounds,
        decay_time_bounds=decay_time_bounds,
        decay_count_bounds=decay_count_bounds,
        lapse_bounds=lapse_bounds,
        x0=x0,
        use_dual_decay=use_dual_decay,
    )
    theta_hat = fit["theta_hat"]
    pred = model.predict(
        data=participant_data,
        lapse_rate=theta_hat["lapse_rate"],
        decay_rate=theta_hat.get("decay_rate"),
        use_dual_decay=theta_hat.get("use_dual_decay", False),
        decay_time=theta_hat.get("decay_time"),
        decay_count=theta_hat.get("decay_count"),
    )
    exp_acc = wm_expected_accuracy(pred, participant_data)
    n = count_nonlost_trials(participant_data)["n_total"]
    k = 3 if theta_hat.get("use_dual_decay", False) else 2
    metrics = aic_bic_from_ll(ll=fit["loglik"], k=k, n=n)
    return {
        "participant_id": participant_id,
        "config": {
            "model": "ExponentialDecayInverseDistanceModel",
            "use_dual_decay": use_dual_decay,
            "fit_decay_bounds": list(decay_bounds),
            "fit_decay_time_bounds": list(decay_time_bounds),
            "fit_decay_count_bounds": list(decay_count_bounds),
            "fit_lapse_bounds": list(lapse_bounds),
        },
        "theta_hat": theta_hat,
        "metrics": metrics,
        "expected_accuracy": exp_acc,
        "participant_accuracy": participant_accuracy(participant_data),
        "fit_info": {
            "success": fit["success"],
            "message": fit["message"],
            "nfev": fit["nfev"],
            "nit": fit["nit"],
            "x0": fit["x0"],
        },
    }


def fit_last_m_model(
    *,
    participant_data: Dict[str, Any],
    participant_id: int,
    seed: int,
    use_category_specific_memory: bool = False,
    M_grid: Iterable[int] = range(1, 13),
    M0_grid: Iterable[int] = range(1, 13),
    M1_grid: Iterable[int] = range(1, 13),
    lapse_bounds: tuple[float, float] = (1e-6, 0.49),
    **_: Any,
) -> Dict[str, Any]:
    from models.WM import LastMInverseDistanceModel
    from utils.metrics import count_nonlost_trials
    from utils.mle_wm import fit_last_m_mle

    rng = np.random.default_rng(seed)
    if use_category_specific_memory:
        model = LastMInverseDistanceModel(
            use_category_specific_memory=True,
            M0=max(int(v) for v in M0_grid),
            M1=max(int(v) for v in M1_grid),
        )
        fit = fit_last_m_mle(
            model=model,
            data=participant_data,
            M0_grid=M0_grid,
            M1_grid=M1_grid,
            lapse_bounds=lapse_bounds,
            use_category_specific_memory=True,
            rng=rng,
        )
    else:
        model = LastMInverseDistanceModel(M=max(int(v) for v in M_grid))
        fit = fit_last_m_mle(
            model=model,
            data=participant_data,
            M_grid=M_grid,
            lapse_bounds=lapse_bounds,
            use_category_specific_memory=False,
            rng=rng,
        )
    theta_hat = fit["theta_hat"]
    pred = model.predict(
        data=participant_data,
        lapse_rate=theta_hat["lapse_rate"],
        M=theta_hat.get("M"),
        use_category_specific_memory=theta_hat.get("use_category_specific_memory", False),
        M0=theta_hat.get("M0"),
        M1=theta_hat.get("M1"),
    )
    exp_acc = wm_expected_accuracy(pred, participant_data)
    n = count_nonlost_trials(participant_data)["n_total"]
    k = 3 if theta_hat.get("use_category_specific_memory", False) else 2
    metrics = aic_bic_from_ll(ll=fit["loglik"], k=k, n=n)
    return {
        "participant_id": participant_id,
        "config": {
            "model": "LastMInverseDistanceModel",
            "use_category_specific_memory": use_category_specific_memory,
            "fit_M_grid": [int(v) for v in M_grid],
            "fit_M0_grid": [int(v) for v in M0_grid],
            "fit_M1_grid": [int(v) for v in M1_grid],
            "fit_lapse_bounds": list(lapse_bounds),
        },
        "theta_hat": theta_hat,
        "metrics": metrics,
        "expected_accuracy": exp_acc,
        "participant_accuracy": participant_accuracy(participant_data),
        "fit_info": {
            "best_run": fit.get("best_run"),
            "n_grid_evals": len(fit.get("all_runs", [])),
        },
    }
