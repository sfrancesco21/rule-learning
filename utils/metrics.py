import numpy as np
import jax.numpy as jnp

def count_nonlost_trials(data) -> dict:
    """Counts non-lost trials in training/test and total."""
    lost_tr = np.asarray(data["training"]["trialLost"]).astype(int)
    lost_te = np.asarray(data["test"]["trialLost"]).astype(int)
    n_tr = int(np.sum(lost_tr == 0))
    n_te = int(np.sum(lost_te == 0))
    return {"n_train": n_tr, "n_test": n_te, "n_total": n_tr + n_te}

def num_free_params_from_model(model, theta_hat: dict) -> int:
    """
    Count free parameters used for fitting.
    For your MAP/MLE routines, these are the constrained parameters in theta_hat.
    """
    k = 0
    for name in ["alpha", "lapse_rate", "sigma_init", "eta_mu", "eta_cov", "rho", "mu_init"]:
        if name in theta_hat:
            arr = np.asarray(theta_hat[name])
            k += int(arr.size) if arr.ndim > 0 else 1
    return int(k)

def loglik_at_theta(model, data, theta_hat: dict) -> float:
    """Evaluate log-likelihood (NOT log posterior) at theta_hat."""
    alpha = jnp.asarray(theta_hat["alpha"])
    lapse = jnp.asarray(theta_hat["lapse_rate"])
    sigma_init = jnp.asarray(theta_hat["sigma_init"]) if "sigma_init" in theta_hat else None
    eta_mu = jnp.asarray(theta_hat["eta_mu"]) if "eta_mu" in theta_hat else None
    eta_cov = jnp.asarray(theta_hat["eta_cov"]) if "eta_cov" in theta_hat else None
    rho = jnp.asarray(theta_hat["rho"]) if "rho" in theta_hat else None
    mu_init = jnp.asarray(theta_hat["mu_init"]) if "mu_init" in theta_hat else None

    ll = model.log_likelihood(
        alpha, lapse, data,
        eta_mu=eta_mu, eta_cov=eta_cov,
        rho=rho,
        mu_init=mu_init,
        sigma_init=sigma_init,
    )
    return float(ll)

def aic_bic_from_mle(model, data, theta_hat: dict) -> dict:
    """
    Compute AIC/BIC using log-likelihood evaluated at MAP parameters.
    Returns AIC, BIC, loglik, n, k.
    """
    n_counts = count_nonlost_trials(data)
    n = n_counts["n_total"]
    if n <= 0:
        raise ValueError("No non-lost trials; cannot compute AIC/BIC.")

    ll = loglik_at_theta(model, data, theta_hat)
    k = num_free_params_from_model(model, theta_hat)

    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    return {
        "loglik": ll,
        "k": k,
        "n": n,
        "AIC": float(aic),
        "BIC": float(bic),
        **n_counts,
    }
