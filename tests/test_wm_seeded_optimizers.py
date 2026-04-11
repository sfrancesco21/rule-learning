import numpy as np

from models.WM import ExponentialDecayInverseDistanceModel, LastMInverseDistanceModel
from utils.mle_wm import fit_exp_decay_mle, fit_last_m_mle


def make_small_data():
    training_stimuli = np.array([[1.0, 1.0], [5.0, 5.0], [1.5, 1.2], [4.8, 5.2]], dtype=float)
    training_labels = np.array([0, 1, 0, 1], dtype=int)
    test_stimuli = np.array([[1.2, 0.9], [5.1, 4.9]], dtype=float)
    test_labels = np.array([0, 1], dtype=int)
    return {
        "training": {
            "trialN": [1, 2, 3, 4],
            "stimuli": training_stimuli,
            "labels": training_labels,
            "outcome": [1, 1, 1, 1],
            "action": training_labels.copy(),
            "trialLost": np.zeros(4, dtype=int),
        },
        "test": {
            "trialN": [5, 6],
            "stimuli": test_stimuli,
            "labels": test_labels,
            "outcome": [1, 1],
            "action": test_labels.copy(),
            "trialLost": np.zeros(2, dtype=int),
        },
    }


def test_seeded_exp_decay_fit_is_reproducible_and_changes_with_seed():
    data = make_small_data()
    model = ExponentialDecayInverseDistanceModel()

    fit_a = fit_exp_decay_mle(model, data, rng=np.random.default_rng(7))
    fit_b = fit_exp_decay_mle(model, data, rng=np.random.default_rng(7))
    fit_c = fit_exp_decay_mle(model, data, rng=np.random.default_rng(8))

    assert fit_a["x0"] == fit_b["x0"]
    assert fit_a["theta_hat"] == fit_b["theta_hat"]
    assert fit_a["x0"] != fit_c["x0"]


def test_seeded_last_m_fit_advances_rng_per_grid_candidate():
    data = make_small_data()
    model = LastMInverseDistanceModel()

    fit_a = fit_last_m_mle(model, data, M_grid=range(1, 4), rng=np.random.default_rng(7))
    fit_b = fit_last_m_mle(model, data, M_grid=range(1, 4), rng=np.random.default_rng(7))
    fit_c = fit_last_m_mle(model, data, M_grid=range(1, 4), rng=np.random.default_rng(8))

    runs_a = fit_a["all_runs"]
    runs_b = fit_b["all_runs"]
    runs_c = fit_c["all_runs"]

    assert [run["M"] for run in runs_a] == [1, 2, 3]
    assert [run["x0_lapse_rate"] for run in runs_a] == [run["x0_lapse_rate"] for run in runs_b]
    assert [run["x0_lapse_rate"] for run in runs_a] != [run["x0_lapse_rate"] for run in runs_c]
