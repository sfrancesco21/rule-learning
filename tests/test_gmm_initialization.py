import jax

from models.GMM import HardLabelTwoMixtureFullCov
from utils.mle import build_unconstrained_init, unpack_theta_from_z


def _unpack_with_seed(model, seed):
    z0, slices = build_unconstrained_init(model, jax.random.PRNGKey(seed))
    alpha, lapse_rate, sigma_init, eta_mu, eta_cov, rho, mu_init = unpack_theta_from_z(model, z0, slices)
    return {
        "alpha": float(alpha),
        "lapse_rate": float(lapse_rate),
        "sigma_init": float(sigma_init),
        "eta_mu": None if eta_mu is None else float(eta_mu),
        "eta_cov": None if eta_cov is None else float(eta_cov),
        "rho": None if rho is None else float(rho),
        "mu_init": [[float(mu_init[0, 0]), float(mu_init[0, 1])], [float(mu_init[1, 0]), float(mu_init[1, 1])]],
    }


def test_gmm_initialization_matches_broad_original_space_bounds():
    model = HardLabelTwoMixtureFullCov(
        config={
            "use_forgetting": True,
            "use_lr_mu": True,
            "use_lr_cov": True,
            "tie_pull_lrs": False,
        }
    )

    theta = _unpack_with_seed(model, seed=7)

    assert 0.1 <= theta["alpha"] <= 8.0
    assert 1e-3 <= theta["lapse_rate"] <= 0.20
    assert 0.25 <= theta["sigma_init"] <= 2.5
    assert 0.80 <= theta["rho"] <= 0.999
    assert 0.02 <= theta["eta_mu"] <= 0.80
    assert 0.02 <= theta["eta_cov"] <= 0.60

    mu0 = theta["mu_init"][0]
    mu1 = theta["mu_init"][1]
    for value in mu0 + mu1:
        assert 0.5 <= value <= 5.5
    assert abs(mu0[0] - mu0[1]) >= 1.5
    assert abs(mu1[0] - mu1[1]) <= 1.25


def test_gmm_initialization_is_reproducible_per_seed_and_changes_across_seeds():
    model = HardLabelTwoMixtureFullCov(
        config={
            "use_forgetting": True,
            "use_lr_mu": True,
            "use_lr_cov": False,
            "tie_pull_lrs": True,
        }
    )

    theta_a = _unpack_with_seed(model, seed=11)
    theta_b = _unpack_with_seed(model, seed=11)
    theta_c = _unpack_with_seed(model, seed=12)

    assert theta_a == theta_b
    assert theta_a != theta_c
