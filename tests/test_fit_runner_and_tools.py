import csv
import json
from pathlib import Path

from build_fit_manifest import build_manifest_rows, write_manifest
from select_best_seed import collect_best_seed_rows, write_best_seed_summary
from utils.fit_runner import run_single_fit
from utils.model_registry import ModelSpec


def test_run_single_fit_writes_seed_scoped_result(monkeypatch, tmp_path):
    def fake_load_participant_data(participant_id, results_dir):
        return {"participant_id": participant_id, "results_dir": str(results_dir)}

    def fake_fit_func(*, participant_data, participant_id, seed, **kwargs):
        return {
            "participant_id": participant_id,
            "config": {"dummy": True},
            "theta_hat": {"seed_value": seed},
            "metrics": {"loglik": -1.0, "AIC": 4.0, "BIC": 5.0},
            "expected_accuracy": {"training": 0.5, "test": 0.5},
            "participant_accuracy": {"training": 0.5, "test": 0.5},
            "fit_info": {"kwargs": kwargs},
        }

    monkeypatch.setattr("utils.fit_runner.load_participant_data", fake_load_participant_data)
    monkeypatch.setattr(
        "utils.fit_runner.get_model_spec",
        lambda _: ModelSpec(name="demo", family="demo", fit_func=fake_fit_func, fit_defaults={"n_starts": 3}),
    )

    status = run_single_fit(
        model_name="demo",
        participant_id=11,
        seed=4,
        output_dir=tmp_path,
        results_dir=tmp_path,
    )

    assert status["status"] == "success"
    results_path = Path(status["results_path"])
    assert results_path.name == "results.json"
    assert results_path.parent.name == "seed_4"

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["model"] == "demo"
    assert payload["participant_id"] == 11
    assert payload["seed"] == 4


def test_run_single_fit_writes_error_payload(monkeypatch, tmp_path):
    monkeypatch.setattr("utils.fit_runner.load_participant_data", lambda participant_id, results_dir: {})

    def broken_fit_func(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "utils.fit_runner.get_model_spec",
        lambda _: ModelSpec(name="demo", family="demo", fit_func=broken_fit_func, fit_defaults={}),
    )

    try:
        run_single_fit(model_name="demo", participant_id=3, seed=9, output_dir=tmp_path, results_dir=tmp_path)
        raise AssertionError("Expected run_single_fit to raise.")
    except RuntimeError:
        pass

    error_path = tmp_path / "demo" / "3" / "seed_9" / "error.json"
    assert error_path.exists()
    payload = json.loads(error_path.read_text(encoding="utf-8"))
    assert payload["error"]["message"] == "boom"


def test_manifest_builder_writes_cross_product(tmp_path):
    rows = build_manifest_rows(["m1", "m2"], [1, 2], [0, 1])
    assert len(rows) == 8

    manifest_path = write_manifest(tmp_path / "fit_manifest.tsv", rows)
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        loaded = list(reader)
    assert len(loaded) == 8
    assert loaded[0] == {"model": "m1", "participant_id": "1", "seed": "0"}


def test_best_seed_selector_chooses_by_requested_criterion(tmp_path):
    base = tmp_path / "model_fits"
    records = [
        ("demo", 1, 0, {"loglik": -10.0, "AIC": 40.0, "BIC": 50.0}),
        ("demo", 1, 1, {"loglik": -8.0, "AIC": 42.0, "BIC": 52.0}),
        ("demo", 2, 0, {"loglik": -12.0, "AIC": 48.0, "BIC": 60.0}),
    ]
    for model_name, participant_id, seed, metrics in records:
        run_dir = base / model_name / str(participant_id) / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model_name,
            "participant_id": participant_id,
            "seed": seed,
            "metrics": metrics,
        }
        (run_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = collect_best_seed_rows(base, criterion="loglik")
    assert len(rows) == 2
    best_for_one = next(row for row in rows if row["participant_id"] == 1)
    assert best_for_one["best_seed"] == 1

    summary_path = write_best_seed_summary(tmp_path / "best_seed_summary.csv", rows)
    assert Path(summary_path).exists()
