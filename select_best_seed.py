from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _score(entry: dict[str, Any], criterion: str) -> float:
    metrics = entry["metrics"]
    if criterion == "loglik":
        return float(metrics["loglik"])
    return -float(metrics[criterion])


def collect_best_seed_rows(fits_dir: str | Path, criterion: str = "loglik") -> list[dict[str, Any]]:
    base = Path(fits_dir).resolve()
    rows: list[dict[str, Any]] = []
    for model_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        for participant_dir in sorted([d for d in model_dir.iterdir() if d.is_dir()], key=lambda p: int(p.name)):
            candidates = []
            for seed_dir in sorted([d for d in participant_dir.iterdir() if d.is_dir()]):
                results_path = seed_dir / "results.json"
                if not results_path.exists():
                    continue
                with results_path.open("r", encoding="utf-8") as f:
                    result = json.load(f)
                if "metrics" not in result:
                    continue
                candidates.append(result | {"results_path": str(results_path.resolve())})
            if not candidates:
                continue
            best = max(candidates, key=lambda entry: _score(entry, criterion))
            rows.append(
                {
                    "model": model_dir.name,
                    "participant_id": int(best["participant_id"]),
                    "best_seed": int(best["seed"]),
                    "criterion": criterion,
                    "loglik": float(best["metrics"]["loglik"]),
                    "AIC": float(best["metrics"]["AIC"]),
                    "BIC": float(best["metrics"]["BIC"]),
                    "results_path": best["results_path"],
                    "n_completed_seeds": len(candidates),
                }
            )
    return rows


def write_best_seed_summary(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "participant_id",
                "best_seed",
                "criterion",
                "loglik",
                "AIC",
                "BIC",
                "results_path",
                "n_completed_seeds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick the best seed per model/participant from raw seed runs.")
    parser.add_argument("--fits-dir", type=str, default="model_fits")
    parser.add_argument("--criterion", choices=["loglik", "AIC", "BIC"], default="loglik")
    parser.add_argument("--output", type=str, default="model_fits/best_seed_summary.csv")
    args = parser.parse_args()

    rows = collect_best_seed_rows(args.fits_dir, args.criterion)
    if not rows:
        raise ValueError(f"No completed results were found under {Path(args.fits_dir).resolve()}.")
    output_path = write_best_seed_summary(args.output, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
