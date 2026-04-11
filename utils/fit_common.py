from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np


def to_py(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, range):
        return list(obj)
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(to_py(payload), f, indent=2)


def wm_expected_accuracy(pred: Dict[str, np.ndarray], data: Dict[str, Any]) -> Dict[str, float]:
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


def participant_accuracy(data: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for phase in ("training", "test"):
        labels = np.asarray(data[phase]["labels"]).astype(int)
        actions = np.asarray(data[phase]["action"]).astype(int)
        lost = np.asarray(data[phase]["trialLost"]).astype(int)
        keep = lost == 0
        out[phase] = float(np.mean(actions[keep] == labels[keep])) if np.any(keep) else float("nan")
    return out


def aic_bic_from_ll(ll: float, k: int, n: int) -> Dict[str, float]:
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return {"loglik": float(ll), "k": int(k), "n": int(n), "AIC": float(aic), "BIC": float(bic)}
