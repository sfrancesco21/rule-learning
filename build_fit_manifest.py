from __future__ import annotations

import argparse
import csv
from pathlib import Path

from utils.cli import parse_csv_spec, parse_int_spec
from utils.model_registry import list_model_names


def build_manifest_rows(models: list[str], participants: list[int], seeds: list[int]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for model_name in models:
        for participant_id in participants:
            for seed in seeds:
                rows.append(
                    {
                        "model": model_name,
                        "participant_id": int(participant_id),
                        "seed": int(seed),
                    }
                )
    return rows


def write_manifest(path: str | Path, rows: list[dict[str, int | str]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "participant_id", "seed"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return output_path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a TSV manifest of model/participant/seed runs.")
    parser.add_argument("--participants", type=str, default="1-125")
    parser.add_argument("--models", type=str, default=",".join(list_model_names()))
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--output", type=str, default="slurm/fit_manifest.tsv")
    args = parser.parse_args()

    rows = build_manifest_rows(
        models=parse_csv_spec(args.models),
        participants=parse_int_spec(args.participants),
        seeds=parse_int_spec(args.seeds),
    )
    output_path = write_manifest(args.output, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
