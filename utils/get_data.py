from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy


def config_dict():
    counter = 0
    conf = {}
    label = {}
    for r in range(1, 7):
        for l in range(1, 7):
            counter += 1
            conf[counter] = [r, l]
            label[counter] = int(abs(r - l) <= 1)
    return [conf, label]


def find_results_dir(base_dir: str | os.PathLike[str] | None = None) -> str:
    if base_dir is not None:
        base_path = Path(base_dir).expanduser()
        candidates = []
        if base_path.is_absolute():
            candidates.append(base_path.resolve())
        else:
            candidates.extend(
                [
                    base_path.resolve(),
                    (Path(__file__).resolve().parent / base_path).resolve(),
                    (Path(__file__).resolve().parents[1] / base_path).resolve(),
                ]
            )
        for candidate in candidates:
            if candidate.is_dir():
                return str(candidate)
        raise FileNotFoundError(f"Results directory not found. Tried: {', '.join(str(p) for p in candidates)}")

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "Results",
        root.parent / "Results",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.resolve())
    raise FileNotFoundError("Could not find Results directory. Tried ./Results and ../Results.")


def _load_phase_data(mat_data: Dict[str, Any], conf) -> Dict[str, Any]:
    results = mat_data["Results"][0][0][0][0]
    n_trials = results.shape[0]
    return {
        "trialN": [results[i][0][0][0] for i in range(n_trials)],
        "stimuli": np.array([conf[0][results[i][1][0][0]] for i in range(n_trials)]),
        "labels": np.array([conf[1][results[i][1][0][0]] for i in range(n_trials)]),
        "outcome": [results[i][2][0][0] for i in range(n_trials)],
        "action": np.array([results[i][4][0][0] for i in range(n_trials)]),
        "trialLost": np.array([results[i][6][0][0] for i in range(n_trials)]),
    }


def load_participant_data(participant_id: int, base_dir: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    conf = config_dict()
    resolved_base_dir = find_results_dir(base_dir)
    folder_name = f"sbj{int(participant_id)}"
    folder_path = os.path.join(resolved_base_dir, folder_name)
    training_pattern = os.path.join(folder_path, f"{folder_name}Training*")
    test_pattern = os.path.join(folder_path, f"{folder_name}Test*")

    training_files = glob.glob(training_pattern)
    test_files = glob.glob(test_pattern)
    if not training_files or not test_files:
        raise FileNotFoundError(f"Files for {folder_name} not found in {resolved_base_dir}.")

    training_data = scipy.io.loadmat(training_files[0])
    test_data = scipy.io.loadmat(test_files[0])
    return {
        "training": _load_phase_data(training_data, conf),
        "test": _load_phase_data(test_data, conf),
    }


def read_mat_files(base_dir: str | os.PathLike[str] | None = None):
    resolved_base_dir = find_results_dir(base_dir)
    data = {}
    for participant_id in range(1, 126):
        try:
            data[participant_id] = load_participant_data(participant_id, resolved_base_dir)
        except FileNotFoundError:
            print(f"Files for sbj{participant_id} not found in {resolved_base_dir}.")
    return data


def participant_accuracy_by_phase(data):
    """
    Returns participant accuracy for training and test, excluding trialLost==1.
    Assumes:
      - data[phase]["labels"] are correct (0/1)
      - data[phase]["action"] are participant actions (0/1)
      - data[phase]["trialLost"] is 0/1
    """
    out = {}
    for phase in ("training", "test"):
        lab = np.asarray(data[phase]["labels"]).astype(int)
        act = np.asarray(data[phase]["action"]).astype(int)
        lost = np.asarray(data[phase]["trialLost"]).astype(int)

        keep = (lost == 0)
        if keep.sum() == 0:
            out[phase] = np.nan
        else:
            out[phase] = float(np.mean(act[keep] == lab[keep]))
    return out

def get_group(s):
    if s % 2 == 0 and s <= 60:
        group = 'blocked1'
    elif s <= 60:
        group = 'interleaved'
    elif s <=90:
        group = 'blocked2'
    else:
        group = 'blocked3'
    return group
