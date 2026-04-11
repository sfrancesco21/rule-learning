from __future__ import annotations

from typing import Any, Dict

import numpy as np

from utils.fit_common import participant_accuracy

HardLabelTwoMixtureFullCov = None
fit_point_estimate_multistart = None
aic_bic_from_mle = None


def _get_gmm_model_cls():
    global HardLabelTwoMixtureFullCov
    if HardLabelTwoMixtureFullCov is None:
        from models.GMM import HardLabelTwoMixtureFullCov as model_cls

        HardLabelTwoMixtureFullCov = model_cls
    return HardLabelTwoMixtureFullCov


def _get_fit_point_estimate_multistart():
    global fit_point_estimate_multistart
    if fit_point_estimate_multistart is None:
        from utils.mle import fit_point_estimate_multistart as fit_fn

        fit_point_estimate_multistart = fit_fn
    return fit_point_estimate_multistart


def _get_aic_bic_from_mle():
    global aic_bic_from_mle
    if aic_bic_from_mle is None:
        from utils.metrics import aic_bic_from_mle as metrics_fn

        aic_bic_from_mle = metrics_fn
    return aic_bic_from_mle


def fit_gmm_model(
    *,
    participant_data: Dict[str, Any],
    participant_id: int,
    seed: int,
    config: Dict[str, Any],
    n_starts: int = 3,
    num_steps: int = 4000,
    lr: float = 5e-3,
    **_: Any,
) -> Dict[str, Any]:
    model = _get_gmm_model_cls()(config)
    theta_hat, _, _, all_runs = _get_fit_point_estimate_multistart()(
        model=model,
        data=participant_data,
        n_starts=n_starts,
        rng_seed=seed,
        num_steps=num_steps,
        lr=lr,
        map_estimate=False,
        print_every=max(200, num_steps),
    )
    metrics = _get_aic_bic_from_mle()(model=model, data=participant_data, theta_hat=theta_hat)
    trace = model.trace_sufficient_stats(
        alpha=theta_hat["alpha"],
        lapse_rate=theta_hat["lapse_rate"],
        data=participant_data,
        eta_mu=theta_hat.get("eta_mu"),
        eta_cov=theta_hat.get("eta_cov"),
        rho=theta_hat.get("rho"),
        mu_init=theta_hat.get("mu_init"),
        sigma_init=theta_hat.get("sigma_init"),
    )
    exp_acc = {
        "training": float(np.nanmean(np.asarray(trace["training"]["expected_acc"], dtype=float))),
        "test": float(np.nanmean(np.asarray(trace["test"]["expected_acc"], dtype=float))),
    }
    multistart_runs = [
        {
            "seed": int(run["seed"]),
            "final_loss": float(run["final_loss"]),
            "loss_history": list(run["history"].get("loss", [])),
        }
        for run in all_runs
    ]
    best_idx = int(np.argmin([run["final_loss"] for run in all_runs]))
    return {
        "participant_id": participant_id,
        "config": config,
        "theta_hat": theta_hat,
        "metrics": metrics,
        "expected_accuracy": exp_acc,
        "participant_accuracy": participant_accuracy(participant_data),
        "fit_info": {
            "n_starts": int(n_starts),
            "multistart_seeds": [int(seed + i) for i in range(n_starts)],
            "best_multistart_index": best_idx,
            "best_final_loss": float(all_runs[best_idx]["final_loss"]),
            "multistart_runs": multistart_runs,
        },
    }
