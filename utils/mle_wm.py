from __future__ import annotations

from typing import Dict, Any, Iterable, Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return float(np.log(p / (1.0 - p)))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def fit_last_m_mle(
    model,
    data: Dict[str, Any],
    M_grid: Optional[Iterable[int]] = None,
    M0_grid: Optional[Iterable[int]] = None,
    M1_grid: Optional[Iterable[int]] = None,
    lapse_bounds=(1e-6, 0.49),
    use_category_specific_memory: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    MLE fit for LastMInverseDistanceModel.
    - M is discrete: searched over M_grid
    - lapse_rate is continuous in bounds, optimized per M
    """
    use_cat = (
        bool(getattr(model, "use_category_specific_memory", False))
        if use_category_specific_memory is None
        else bool(use_category_specific_memory)
    )

    if M_grid is None:
        M_grid = range(1, 81)
    if M0_grid is None:
        M0_grid = range(0, 81)
    if M1_grid is None:
        M1_grid = range(0, 81)

    best = None
    all_runs = []

    if not use_cat:
        for M in M_grid:
            M_int = int(M)
            if M_int < 1:
                continue

            def objective_lapse(lapse):
                ll = model.log_likelihood(
                    data=data,
                    lapse_rate=float(lapse),
                    M=M_int,
                    use_category_specific_memory=False,
                )
                return -float(ll)

            out = minimize_scalar(
                objective_lapse,
                method="bounded",
                bounds=(float(lapse_bounds[0]), float(lapse_bounds[1])),
                options={"xatol": 1e-4},
            )

            lapse_hat = float(out.x)
            ll_hat = -float(out.fun)
            run = {
                "M": M_int,
                "lapse_rate": lapse_hat,
                "loglik": ll_hat,
                "success": bool(out.success),
                "nit": int(getattr(out, "nit", -1)),
                "use_category_specific_memory": False,
            }
            all_runs.append(run)
            if best is None or run["loglik"] > best["loglik"]:
                best = run
        theta_hat = {
            "M": best["M"],
            "lapse_rate": best["lapse_rate"],
            "use_category_specific_memory": False,
        }
    else:
        for M0 in M0_grid:
            M0_int = int(M0)
            if M0_int < 0:
                continue
            for M1 in M1_grid:
                M1_int = int(M1)
                if M1_int < 0:
                    continue

                def objective_lapse(lapse):
                    ll = model.log_likelihood(
                        data=data,
                        lapse_rate=float(lapse),
                        use_category_specific_memory=True,
                        M0=M0_int,
                        M1=M1_int,
                    )
                    return -float(ll)

                out = minimize_scalar(
                    objective_lapse,
                    method="bounded",
                    bounds=(float(lapse_bounds[0]), float(lapse_bounds[1])),
                    options={"xatol": 1e-4},
                )

                lapse_hat = float(out.x)
                ll_hat = -float(out.fun)
                run = {
                    "M0": M0_int,
                    "M1": M1_int,
                    "lapse_rate": lapse_hat,
                    "loglik": ll_hat,
                    "success": bool(out.success),
                    "nit": int(getattr(out, "nit", -1)),
                    "use_category_specific_memory": True,
                }
                all_runs.append(run)
                if best is None or run["loglik"] > best["loglik"]:
                    best = run
        theta_hat = {
            "M0": best["M0"],
            "M1": best["M1"],
            "lapse_rate": best["lapse_rate"],
            "use_category_specific_memory": True,
        }

    return {
        "theta_hat": theta_hat,
        "loglik": best["loglik"],
        "best_run": best,
        "all_runs": all_runs,
    }


def fit_exp_decay_mle(
    model,
    data: Dict[str, Any],
    decay_bounds=(1e-6, 10.0),
    decay_time_bounds=(1e-6, 10.0),
    decay_count_bounds=(1e-6, 10.0),
    lapse_bounds=(1e-6, 0.49),
    x0: Optional[Dict[str, float]] = None,
    use_dual_decay: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    MLE fit for ExponentialDecayInverseDistanceModel.
    Optimizes (decay_rate, lapse_rate) jointly via unconstrained reparameterization.
    """
    use_dual = (
        bool(getattr(model, "use_dual_decay", False))
        if use_dual_decay is None
        else bool(use_dual_decay)
    )

    if x0 is None:
        x0 = {"lapse_rate": 0.05}
    l_lo, l_hi = float(lapse_bounds[0]), float(lapse_bounds[1])
    l_scale = max(l_hi - l_lo, 1e-12)

    if not use_dual:
        if "decay_rate" not in x0:
            x0["decay_rate"] = float(getattr(model, "decay_rate", 0.1))
        d_lo, d_hi = float(decay_bounds[0]), float(decay_bounds[1])
        d_scale = max(d_hi - d_lo, 1e-12)
        d0_u = (float(x0["decay_rate"]) - d_lo) / d_scale
        l0_u = (float(x0["lapse_rate"]) - l_lo) / l_scale
        z0 = np.array([_logit(d0_u), _logit(l0_u)], dtype=float)

        def unpack(z):
            d = d_lo + d_scale * _sigmoid(float(z[0]))
            l = l_lo + l_scale * _sigmoid(float(z[1]))
            return float(d), float(l)

        def objective(z):
            decay_rate, lapse_rate = unpack(z)
            ll = model.log_likelihood(
                data=data,
                lapse_rate=lapse_rate,
                decay_rate=decay_rate,
                use_dual_decay=False,
            )
            val = -float(ll)
            if not np.isfinite(val):
                return 1e30
            return val

        out = minimize(objective, z0, method="L-BFGS-B")
        decay_hat, lapse_hat = unpack(out.x)
        ll_hat = -float(out.fun)
        theta_hat = {
            "decay_rate": decay_hat,
            "lapse_rate": lapse_hat,
            "use_dual_decay": False,
        }
    else:
        if "decay_time" not in x0:
            x0["decay_time"] = float(getattr(model, "decay_time", 0.1))
        if "decay_count" not in x0:
            x0["decay_count"] = float(getattr(model, "decay_count", 0.1))

        dt_lo, dt_hi = float(decay_time_bounds[0]), float(decay_time_bounds[1])
        dc_lo, dc_hi = float(decay_count_bounds[0]), float(decay_count_bounds[1])
        dt_scale = max(dt_hi - dt_lo, 1e-12)
        dc_scale = max(dc_hi - dc_lo, 1e-12)

        dt0_u = (float(x0["decay_time"]) - dt_lo) / dt_scale
        dc0_u = (float(x0["decay_count"]) - dc_lo) / dc_scale
        l0_u = (float(x0["lapse_rate"]) - l_lo) / l_scale
        z0 = np.array([_logit(dt0_u), _logit(dc0_u), _logit(l0_u)], dtype=float)

        def unpack(z):
            dt = dt_lo + dt_scale * _sigmoid(float(z[0]))
            dc = dc_lo + dc_scale * _sigmoid(float(z[1]))
            l = l_lo + l_scale * _sigmoid(float(z[2]))
            return float(dt), float(dc), float(l)

        def objective(z):
            decay_time, decay_count, lapse_rate = unpack(z)
            ll = model.log_likelihood(
                data=data,
                lapse_rate=lapse_rate,
                use_dual_decay=True,
                decay_time=decay_time,
                decay_count=decay_count,
            )
            val = -float(ll)
            if not np.isfinite(val):
                return 1e30
            return val

        out = minimize(objective, z0, method="L-BFGS-B")
        decay_time_hat, decay_count_hat, lapse_hat = unpack(out.x)
        ll_hat = -float(out.fun)
        theta_hat = {
            "decay_time": decay_time_hat,
            "decay_count": decay_count_hat,
            "lapse_rate": lapse_hat,
            "use_dual_decay": True,
        }

    return {
        "theta_hat": theta_hat,
        "loglik": ll_hat,
        "success": bool(out.success),
        "message": str(out.message),
        "nfev": int(getattr(out, "nfev", -1)),
        "nit": int(getattr(out, "nit", -1)),
    }
