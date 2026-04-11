from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "agents"))
sys.path.append(os.path.join(ROOT, "utils"))

from GMM import HardLabelTwoMixtureFullCov
from WM import LastMInverseDistanceModel, ExponentialDecayInverseDistanceModel
from get_data import read_mat_files
from metrics import aic_bic_from_mle, count_nonlost_trials
from mle import fit_point_estimate_multistart
from mle_wm import fit_last_m_mle, fit_exp_decay_mle


def _to_py(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _wm_expected_accuracy(pred: Dict[str, np.ndarray], data: Dict[str, Any]) -> Dict[str, float]:
    y_tr = np.asarray(data["training"]["labels"]).astype(float)
    y_te = np.asarray(data["test"]["labels"]).astype(float)
    p1_tr = np.asarray(pred["p1_train"], dtype=float)
    p1_te = np.asarray(pred["p1_test"], dtype=float)

    exp_acc_tr = y_tr * p1_tr + (1.0 - y_tr) * (1.0 - p1_tr)
    exp_acc_te = y_te * p1_te + (1.0 - y_te) * (1.0 - p1_te)

    return {
        "training": float(np.nanmean(exp_acc_tr)),
        "test": float(np.nanmean(exp_acc_te)),
    }


def _participant_accuracy(data: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for phase in ("training", "test"):
        labels = np.asarray(data[phase]["labels"]).astype(int)
        actions = np.asarray(data[phase]["action"]).astype(int)
        lost = np.asarray(data[phase]["trialLost"]).astype(int)
        keep = lost == 0
        out[phase] = float(np.mean(actions[keep] == labels[keep])) if np.any(keep) else float("nan")
    return out


def _aic_bic_from_ll(ll: float, k: int, n: int) -> Dict[str, float]:
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return {"loglik": float(ll), "k": int(k), "n": int(n), "AIC": float(aic), "BIC": float(bic)}


def _fit_gmm_model(
    participant_data: Dict[str, Any],
    participant_id: int,
    cfg: Dict[str, Any],
    n_starts: int,
    num_steps: int,
    lr: float,
) -> Dict[str, Any]:
    model = HardLabelTwoMixtureFullCov(cfg)
    theta_hat, _, _, _ = fit_point_estimate_multistart(
        model=model,
        data=participant_data,
        n_starts=n_starts,
        rng_seed=participant_id,
        num_steps=num_steps,
        lr=lr,
        map_estimate=False,
        print_every=max(200, num_steps),
    )
    metrics = aic_bic_from_mle(model=model, data=participant_data, theta_hat=theta_hat)
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
    return {
        "participant_id": participant_id,
        "config": cfg,
        "theta_hat": theta_hat,
        "metrics": metrics,
        "expected_accuracy": exp_acc,
        "participant_accuracy": _participant_accuracy(participant_data),
    }


def _fit_exp_decay_model(
    participant_data: Dict[str, Any],
    participant_id: int,
    use_dual_decay: bool = False,
) -> Dict[str, Any]:
    if use_dual_decay:
        model = ExponentialDecayInverseDistanceModel(
            use_dual_decay=True,
            decay_time=0.1,
            decay_count=0.1,
        )
    else:
        model = ExponentialDecayInverseDistanceModel(decay_rate=0.1)
    fit = fit_exp_decay_mle(
        model=model,
        data=participant_data,
        decay_bounds=(1e-6, 10.0),
        decay_time_bounds=(1e-6, 10.0),
        decay_count_bounds=(1e-6, 10.0),
        lapse_bounds=(1e-6, 0.49),
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
    exp_acc = _wm_expected_accuracy(pred, participant_data)
    n = count_nonlost_trials(participant_data)["n_total"]
    k = 3 if theta_hat.get("use_dual_decay", False) else 2
    metrics = _aic_bic_from_ll(ll=fit["loglik"], k=k, n=n)

    return {
        "participant_id": participant_id,
        "config": {
            "model": "ExponentialDecayInverseDistanceModel",
            "use_dual_decay": use_dual_decay,
            "fit_decay_bounds": [1e-6, 10.0],
            "fit_decay_time_bounds": [1e-6, 10.0],
            "fit_decay_count_bounds": [1e-6, 10.0],
            "fit_lapse_bounds": [1e-6, 0.49],
        },
        "theta_hat": theta_hat,
        "metrics": metrics,
        "expected_accuracy": exp_acc,
        "participant_accuracy": _participant_accuracy(participant_data),
        "fit_info": {
            "success": fit["success"],
            "message": fit["message"],
            "nfev": fit["nfev"],
            "nit": fit["nit"],
        },
    }


def _fit_last_m_model(
    participant_data: Dict[str, Any],
    participant_id: int,
    use_category_specific_memory: bool = False,
) -> Dict[str, Any]:
    if use_category_specific_memory:
        model = LastMInverseDistanceModel(
            use_category_specific_memory=True,
            M0=12,
            M1=12,
        )
        fit = fit_last_m_mle(
            model=model,
            data=participant_data,
            M0_grid=range(1, 13),
            M1_grid=range(1, 13),
            lapse_bounds=(1e-6, 0.49),
            use_category_specific_memory=True,
        )
    else:
        model = LastMInverseDistanceModel(M=12)
        fit = fit_last_m_mle(
            model=model,
            data=participant_data,
            M_grid=range(1, 13),
            lapse_bounds=(1e-6, 0.49),
            use_category_specific_memory=False,
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
    exp_acc = _wm_expected_accuracy(pred, participant_data)
    n = count_nonlost_trials(participant_data)["n_total"]
    k = 3 if theta_hat.get("use_category_specific_memory", False) else 2
    metrics = _aic_bic_from_ll(ll=fit["loglik"], k=k, n=n)

    return {
        "participant_id": participant_id,
        "config": {
            "model": "LastMInverseDistanceModel",
            "use_category_specific_memory": use_category_specific_memory,
            "fit_M_grid": list(range(1, 13)),
            "fit_M0_grid": list(range(1, 13)),
            "fit_M1_grid": list(range(1, 13)),
            "fit_lapse_bounds": [1e-6, 0.49],
        },
        "theta_hat": theta_hat,
        "metrics": metrics,
        "expected_accuracy": exp_acc,
        "participant_accuracy": _participant_accuracy(participant_data),
        "fit_info": {"best_run": fit.get("best_run")},
    }


def _find_results_dir() -> str:
    candidates = [
        os.path.join(ROOT, "Results"),
        os.path.join(ROOT, "..", "Results"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return os.path.abspath(p)
    raise FileNotFoundError("Could not find Results directory. Tried ./Results and ../Results.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit GMM/WM models for participants 1..60 and save results.")
    parser.add_argument("--output-dir", type=str, default="model_fits")
    parser.add_argument("--num-steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--n-starts", type=int, default=3)
    args = parser.parse_args()

    results_dir = _find_results_dir()
    data = read_mat_files(base_dir=results_dir)

    gmm_cfgs = {
        # "gmm_no_forgetting_no_lr":{
        #     "K0": 3, "K1": 3, "eps_cov": 1e-3,
        #     "use_forgetting": False,
        #     "use_lr_mu": False,
        #     "use_lr_cov": False,
        #     "tie_pull_lrs": False,
        # },
        # "gmm_no_forgetting_tied_lrs": {
        #     "K0": 3, "K1": 3, "eps_cov": 1e-3,
        #     "use_forgetting": False,
        #     "use_lr_mu": True,
        #     "use_lr_cov": True,
        #     "tie_pull_lrs": True,
        # },
        # "gmm_forgetting_no_lr": {
        #     "K0": 3, "K1": 3, "eps_cov": 1e-3,
        #     "use_forgetting": True,
        #     "use_lr_mu": False,
        #     "use_lr_cov": False,
        # },
        # "gmm_forgetting_tied_lrs": {
        #     "K0": 3, "K1": 3, "eps_cov": 1e-3,
        #     "use_forgetting": True,
        #     "use_lr_mu": True,
        #     "use_lr_cov": True,
        #     "tie_pull_lrs": True,
        # },
    }

    output_dir = os.path.join(ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    errors = []

    run_meta: Dict[str, Any] = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "participants": [61, 120],
        "num_steps": args.num_steps,
        "lr": args.lr,
        "n_starts": args.n_starts,
        "wm_last_m_grid": [1, 12],
        "wm_last_m_category_specific_grids": {"M0": [1, 12], "M1": [1, 12]},
        "wm_exp_decay_bounds": [1e-6, 10.0],
        "wm_exp_decay_dual_bounds": {"decay_time": [1e-6, 10.0], "decay_count": [1e-6, 10.0]},
        "results_dir": output_dir,
    }

    for pid in range(1, 126):
        if pid not in data:
            errors.append({"participant_id": pid, "model": "all", "error": "missing data"})
            continue
        pdata = data[pid]
        print(f"Participant {pid}/125")

        for model_name, cfg in gmm_cfgs.items():
            try:
                res = _fit_gmm_model(
                    participant_data=pdata,
                    participant_id=pid,
                    cfg=cfg,
                    n_starts=args.n_starts,
                    num_steps=args.num_steps,
                    lr=args.lr,
                )
                model_dir = os.path.join(output_dir, model_name, str(pid))
                os.makedirs(model_dir, exist_ok=True)
                with open(os.path.join(model_dir, "results.json"), "w", encoding="utf-8") as f:
                    json.dump(_to_py(res), f, indent=2)
            except Exception as e:
                errors.append({"participant_id": pid, "model": model_name, "error": str(e)})

        # try:
        #     res_decay = _fit_exp_decay_model(
        #         participant_data=pdata,
        #         participant_id=pid,
        #         use_dual_decay=False,
        #     )
        #     model_name = "wm_exponential_decay"
        #     model_dir = os.path.join(output_dir, model_name, str(pid))
        #     os.makedirs(model_dir, exist_ok=True)
        #     with open(os.path.join(model_dir, "results.json"), "w", encoding="utf-8") as f:
        #         json.dump(_to_py(res_decay), f, indent=2)
        # except Exception as e:
        #     errors.append({"participant_id": pid, "model": "wm_exponential_decay", "error": str(e)})

        try:
            res_decay_dual = _fit_exp_decay_model(
                participant_data=pdata,
                participant_id=pid,
                use_dual_decay=True,
            )
            model_name = "wm_exponential_decay_dual"
            model_dir = os.path.join(output_dir, model_name, str(pid))
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(_to_py(res_decay_dual), f, indent=2)
        except Exception as e:
            errors.append({"participant_id": pid, "model": "wm_exponential_decay_dual", "error": str(e)})

        # try:
        #     res_lastm = _fit_last_m_model(
        #         participant_data=pdata,
        #         participant_id=pid,
        #         use_category_specific_memory=False,
        #     )
        #     model_name = "wm_last_m"
        #     model_dir = os.path.join(output_dir, model_name, str(pid))
        #     os.makedirs(model_dir, exist_ok=True)
        #     with open(os.path.join(model_dir, "results.json"), "w", encoding="utf-8") as f:
        #         json.dump(_to_py(res_lastm), f, indent=2)
        # except Exception as e:
        #     errors.append({"participant_id": pid, "model": "wm_last_m", "error": str(e)})

        try:
            res_lastm_cat = _fit_last_m_model(
                participant_data=pdata,
                participant_id=pid,
                use_category_specific_memory=True,
            )
            model_name = "wm_last_m_category_specific"
            model_dir = os.path.join(output_dir, model_name, str(pid))
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(_to_py(res_lastm_cat), f, indent=2)
        except Exception as e:
            errors.append({"participant_id": pid, "model": "wm_last_m_category_specific", "error": str(e)})

    with open(os.path.join(output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(_to_py(run_meta), f, indent=2)
    with open(os.path.join(output_dir, "errors.json"), "w", encoding="utf-8") as f:
        json.dump(_to_py(errors), f, indent=2)

    print(f"Saved results to {output_dir}")
    if errors:
        print(f"Completed with {len(errors)} errors. See {os.path.join(output_dir, 'errors.json')}.")


if __name__ == "__main__":
    main()
