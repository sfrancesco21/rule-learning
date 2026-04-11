from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict

from utils.fit_common import utc_now_iso, write_json
from utils.get_data import find_results_dir, load_participant_data
from utils.model_registry import get_model_spec


def resolve_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    return (Path(__file__).resolve().parents[1] / output_path).resolve()


def build_run_dir(output_dir: str | Path, model_name: str, participant_id: int, seed: int) -> Path:
    return resolve_output_dir(output_dir) / model_name / str(int(participant_id)) / f"seed_{int(seed)}"


def run_single_fit(
    *,
    model_name: str,
    participant_id: int,
    seed: int,
    output_dir: str | Path = "model_fits",
    results_dir: str | Path | None = None,
    overwrite: bool = False,
    **fit_overrides: Any,
) -> Dict[str, Any]:
    spec = get_model_spec(model_name)
    resolved_results_dir = Path(find_results_dir(results_dir))
    run_dir = build_run_dir(output_dir, model_name, participant_id, seed)
    results_path = run_dir / "results.json"
    error_path = run_dir / "error.json"

    if results_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "model": model_name,
            "participant_id": int(participant_id),
            "seed": int(seed),
            "results_path": str(results_path.resolve()),
        }

    fit_options = dict(spec.fit_defaults)
    fit_options.update({k: v for k, v in fit_overrides.items() if v is not None})

    run_info = {
        "created_at_utc": utc_now_iso(),
        "source_results_dir": str(resolved_results_dir.resolve()),
        "source_participant_dir": str((resolved_results_dir / f"sbj{int(participant_id)}").resolve()),
        "output_dir": str(resolve_output_dir(output_dir)),
        "fit_options": fit_options,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        participant_data = load_participant_data(participant_id, resolved_results_dir)
        result = spec.fit_func(
            participant_data=participant_data,
            participant_id=participant_id,
            seed=seed,
            **fit_options,
        )
        result["model"] = model_name
        result["participant_id"] = int(participant_id)
        result["seed"] = int(seed)
        result["run_info"] = run_info
        write_json(results_path, result)
        if error_path.exists():
            error_path.unlink()
        return {
            "status": "success",
            "model": model_name,
            "participant_id": int(participant_id),
            "seed": int(seed),
            "results_path": str(results_path.resolve()),
        }
    except Exception as exc:
        error_payload = {
            "model": model_name,
            "participant_id": int(participant_id),
            "seed": int(seed),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "run_info": run_info,
        }
        write_json(error_path, error_payload)
        raise
