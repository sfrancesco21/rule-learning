import numpy as np

from utils.fit_gmm import fit_gmm_model


class FakeGMM:
    def __init__(self, config):
        self.config = config

    def trace_sufficient_stats(self, **kwargs):
        return {
            "training": {"expected_acc": np.array([0.2, 0.4])},
            "test": {"expected_acc": np.array([0.6, 0.8])},
        }


def test_gmm_fit_uses_base_seed_for_multistarts(monkeypatch):
    def fake_fit_point_estimate_multistart(*, n_starts, rng_seed, **kwargs):
        theta_hat = {
            "alpha": float(rng_seed),
            "lapse_rate": 0.1,
            "sigma_init": 1.0,
            "mu_init": [[1.0, 1.0], [2.0, 2.0]],
        }
        all_runs = [
            {"seed": rng_seed + i, "history": {"loss": [float(rng_seed + i)]}, "final_loss": float(rng_seed + i)}
            for i in range(n_starts)
        ]
        return theta_hat, None, None, all_runs

    monkeypatch.setattr("utils.fit_gmm.HardLabelTwoMixtureFullCov", FakeGMM)
    monkeypatch.setattr("utils.fit_gmm.fit_point_estimate_multistart", fake_fit_point_estimate_multistart)
    monkeypatch.setattr(
        "utils.fit_gmm.aic_bic_from_mle",
        lambda model, data, theta_hat: {"loglik": -1.0, "AIC": 1.0, "BIC": 2.0},
    )

    data = {
        "training": {"labels": np.array([0]), "action": np.array([0]), "trialLost": np.array([0])},
        "test": {"labels": np.array([1]), "action": np.array([1]), "trialLost": np.array([0])},
    }

    fit_a = fit_gmm_model(participant_data=data, participant_id=5, seed=10, config={"demo": True})
    fit_b = fit_gmm_model(participant_data=data, participant_id=5, seed=10, config={"demo": True})
    fit_c = fit_gmm_model(participant_data=data, participant_id=5, seed=11, config={"demo": True})

    assert fit_a["theta_hat"] == fit_b["theta_hat"]
    assert fit_a["theta_hat"] != fit_c["theta_hat"]
    assert fit_a["fit_info"]["multistart_seeds"] == [10, 11, 12]
