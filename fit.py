from __future__ import annotations

import argparse
import sys

from utils.fit_runner import run_single_fit
from utils.model_registry import list_model_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit one model for one participant and one seed.")
    parser.add_argument("--model", type=str, required=True, choices=list_model_names())
    parser.add_argument("--participant-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="model_fits")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        status = run_single_fit(
            model_name=args.model,
            participant_id=args.participant_id,
            seed=args.seed,
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            num_steps=args.num_steps,
            lr=args.lr,
            n_starts=args.n_starts,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"Fit failed for model={args.model} participant={args.participant_id} seed={args.seed}: {exc}", file=sys.stderr)
        return 1

    print(f"{status['status']}: {status['results_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
