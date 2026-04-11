from __future__ import annotations

import argparse
import sys

from utils.cli import parse_csv_spec, parse_int_spec
from utils.fit_runner import run_single_fit
from utils.model_registry import list_model_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential local wrapper over fit.py-style single-run fits.")
    parser.add_argument("--participants", type=str, default="1-125")
    parser.add_argument("--models", type=str, default=",".join(list_model_names()))
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--output-dir", type=str, default="model_fits")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    participants = parse_int_spec(args.participants)
    models = parse_csv_spec(args.models)
    seeds = parse_int_spec(args.seeds)

    errors = []
    total = len(participants) * len(models) * len(seeds)
    done = 0
    for participant_id in participants:
        for model_name in models:
            for seed in seeds:
                done += 1
                print(f"[{done}/{total}] model={model_name} participant={participant_id} seed={seed}")
                try:
                    status = run_single_fit(
                        model_name=model_name,
                        participant_id=participant_id,
                        seed=seed,
                        output_dir=args.output_dir,
                        results_dir=args.results_dir,
                        num_steps=args.num_steps,
                        lr=args.lr,
                        n_starts=args.n_starts,
                        overwrite=args.overwrite,
                    )
                    print(f"  -> {status['status']}: {status['results_path']}")
                except Exception as exc:
                    errors.append((model_name, participant_id, seed, str(exc)))
                    print(f"  -> error: {exc}", file=sys.stderr)
                    if args.stop_on_error:
                        return 1

    if errors:
        print(f"Completed with {len(errors)} errors.", file=sys.stderr)
        return 1
    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
