import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any
import numpyro.distributions as dist


def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def logit(p):
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)


def _sample_uniform(key, low, high):
    low = jnp.asarray(low, dtype=jnp.float32)
    high = jnp.asarray(high, dtype=jnp.float32)
    return low + (high - low) * jax.random.uniform(key, shape=low.shape, dtype=jnp.float32)


def _bounded_interval(lower, upper, fallback_lower=None, fallback_upper=None):
    lo = float(lower)
    hi = float(upper)
    if fallback_lower is not None:
        lo = max(lo, float(fallback_lower))
    if fallback_upper is not None:
        hi = min(hi, float(fallback_upper))
    if hi <= lo:
        if fallback_lower is None or fallback_upper is None or float(fallback_upper) <= float(fallback_lower):
            raise ValueError(f"Invalid initialization interval [{lo}, {hi}]")
        lo = float(fallback_lower)
        hi = float(fallback_upper)
    return lo, hi


def _sample_mu_pair_with_constraint(key, low, high, predicate, max_attempts=256):
    current_key = key
    last_candidate = None
    for _ in range(max_attempts):
        current_key, sample_key = jax.random.split(current_key)
        candidate = _sample_uniform(sample_key, jnp.array([low, low]), jnp.array([high, high]))
        last_candidate = candidate
        if bool(predicate(candidate)):
            return candidate
    raise RuntimeError(f"Failed to sample a valid initial mean pair; last candidate was {last_candidate}.")


# ---------------------------
# New: build/init/unpack/prior
# ---------------------------

def build_unconstrained_init(model, rng_key):
    """
    Returns:
      z0: (D,) unconstrained parameter vector
      slices: dict[str, slice] mapping names -> indices in z

    Always includes:
      - log_alpha
      - logit_lapse
      - log_sigma_init
    Also includes:
      - logit_mu00, logit_mu01, logit_mu10, logit_mu11 (initial means by label/dimension)
    Optionally includes (depending on model flags):
      - logit_rho
      - pull LRs: logit_eta / logit_eta_mu / logit_eta_cov
    """
    parts = []
    slices = {}
    idx = 0

    def add(name, init_value):
        nonlocal idx
        parts.append(jnp.asarray(init_value).reshape(1,))
        slices[name] = slice(idx, idx + 1)
        idx += 1

    init_bounds = {
        "alpha": (0.1, 8.0),
        "lapse_rate": (1e-3, 0.20),
        "sigma_init": (0.25, 2.5),
        "rho": (0.80, 0.999),
        "eta_mu": (0.02, 0.80),
        "eta_cov": (0.02, 0.60),
        "mu_value": (0.5, 5.5),
    }

    alpha_lo, alpha_hi = init_bounds["alpha"]
    lapse_lo, lapse_hi = init_bounds["lapse_rate"]
    sigma_lo, sigma_hi = _bounded_interval(
        *init_bounds["sigma_init"],
        fallback_lower=getattr(model, "sigma_init_min", 1e-3),
        fallback_upper=getattr(model, "sigma_init_max", 10.0),
    )

    mu_min = float(getattr(model, "mu_min", 0.0))
    mu_max = float(getattr(model, "mu_max", 6.0))
    mu_lo, mu_hi = _bounded_interval(
        *init_bounds["mu_value"],
        fallback_lower=mu_min,
        fallback_upper=mu_max,
    )

    current_key = rng_key

    def next_key():
        nonlocal current_key
        current_key, sample_key = jax.random.split(current_key)
        return sample_key

    alpha = _sample_uniform(next_key(), alpha_lo, alpha_hi)
    add("log_alpha", jnp.log(alpha))

    lapse_rate = _sample_uniform(next_key(), lapse_lo, lapse_hi)
    add("logit_lapse", logit(lapse_rate))

    sigma_init = _sample_uniform(next_key(), sigma_lo, sigma_hi)
    add("log_sigma_init", jnp.log(sigma_init))

    mu_min = float(getattr(model, "mu_min", 0.0))
    mu_scale = max(mu_max - mu_min, 1e-6)
    mu0_init = _sample_mu_pair_with_constraint(
        next_key(),
        mu_lo,
        mu_hi,
        lambda candidate: jnp.abs(candidate[0] - candidate[1]) >= 1.5,
    )
    mu1_init = _sample_mu_pair_with_constraint(
        next_key(),
        mu_lo,
        mu_hi,
        lambda candidate: jnp.abs(candidate[0] - candidate[1]) <= 1.25,
    )
    mu0_unit = (mu0_init - mu_min) / mu_scale
    mu1_unit = (mu1_init - mu_min) / mu_scale
    add("logit_mu00", logit(mu0_unit[0]))
    add("logit_mu01", logit(mu0_unit[1]))
    add("logit_mu10", logit(mu1_unit[0]))
    add("logit_mu11", logit(mu1_unit[1]))

    if getattr(model, "use_forgetting", False):
        rho = _sample_uniform(next_key(), *init_bounds["rho"])
        add("logit_rho", logit(rho))

    # ---- Pull learning rates ----
    use_lr_mu = getattr(model, "use_lr_mu", False)
    use_lr_cov = getattr(model, "use_lr_cov", False)
    tie_pull_lrs = getattr(model, "tie_pull_lrs", False)

    if (use_lr_mu or use_lr_cov):
        if use_lr_mu and use_lr_cov and tie_pull_lrs:
            eta = _sample_uniform(next_key(), *init_bounds["eta_mu"])
            add("logit_eta", logit(eta))
        else:
            if use_lr_mu:
                eta_mu = _sample_uniform(next_key(), *init_bounds["eta_mu"])
                add("logit_eta_mu", logit(eta_mu))
            if use_lr_cov:
                eta_cov = _sample_uniform(next_key(), *init_bounds["eta_cov"])
                add("logit_eta_cov", logit(eta_cov))

    z0 = jnp.concatenate(parts, axis=0) if parts else jnp.zeros((0,), dtype=jnp.float32)
    return z0, slices


def unpack_theta_from_z(model, z, slices):
    """
    Returns:
      alpha, lapse_rate, sigma_init, eta_mu, eta_cov, rho, mu_init

    Any optional parameter not used by the model is returned as None.
    """
    log_alpha = z[slices["log_alpha"]][0]
    alpha = jnp.exp(log_alpha)

    logit_lapse = z[slices["logit_lapse"]][0]
    lapse_rate = sigmoid(logit_lapse)
    sigma_init = jnp.exp(z[slices["log_sigma_init"]][0])
    sigma_min = float(getattr(model, "sigma_init_min", 1e-3))
    sigma_max = float(getattr(model, "sigma_init_max", 10.0))
    sigma_init = jnp.clip(sigma_init, sigma_min, sigma_max)

    mu_min = float(getattr(model, "mu_min", 0.0))
    mu_max = float(getattr(model, "mu_max", 6.0))
    mu_scale = mu_max - mu_min
    mu_init = jnp.array([
        [z[slices["logit_mu00"]][0], z[slices["logit_mu01"]][0]],
        [z[slices["logit_mu10"]][0], z[slices["logit_mu11"]][0]],
    ], dtype=jnp.float32)
    mu_init = mu_min + mu_scale * sigmoid(mu_init)

    rho = None
    if "logit_rho" in slices:
        rho = sigmoid(z[slices["logit_rho"]][0])

    # pull
    eta_mu = None
    eta_cov = None
    if "logit_eta" in slices:
        eta = sigmoid(z[slices["logit_eta"]][0])
        eta_mu = eta
        eta_cov = eta
    else:
        if "logit_eta_mu" in slices:
            eta_mu = sigmoid(z[slices["logit_eta_mu"]][0])
        if "logit_eta_cov" in slices:
            eta_cov = sigmoid(z[slices["logit_eta_cov"]][0])

    return alpha, lapse_rate, sigma_init, eta_mu, eta_cov, rho, mu_init


def log_prior_from_z(model, z, slices):
    """
    Log-prior in terms of model priors, plus Jacobian for alpha=exp(log_alpha).
    Uses Normal(0, prior_logit_sd) for any logit-* rate parameter.
    """
    # alpha prior with Jacobian
    log_alpha = z[slices["log_alpha"]][0]
    alpha = jnp.exp(log_alpha)
    lp = dist.Gamma(model.prior_alpha_shape, model.prior_alpha_rate).log_prob(alpha) + log_alpha

    # lapse prior (on logit)
    logit_lapse = z[slices["logit_lapse"]][0]
    lp = lp + dist.Normal(0.0, model.prior_lapse_logit_sd).log_prob(logit_lapse)
    lp = lp + dist.Normal(jnp.log(getattr(model, "sigma_init", 1.707)), getattr(model, "prior_log_sigma_init_sd", 0.5)).log_prob(
        z[slices["log_sigma_init"]][0]
    )

    # bounded mu_init is represented in unconstrained logit space
    mu_init_prior = jnp.asarray(getattr(model, "mu_init_prior", ((3.0, 3.0), (3.0, 3.0))), dtype=jnp.float32)
    mu_min = float(getattr(model, "mu_min", 0.0))
    mu_max = float(getattr(model, "mu_max", 6.0))
    mu_scale = max(mu_max - mu_min, 1e-6)
    mu_prior_unit = jnp.clip((mu_init_prior - mu_min) / mu_scale, 1e-6, 1.0 - 1e-6)
    mu_logit_prior = logit(mu_prior_unit)

    for name, loc in [
        ("logit_mu00", mu_logit_prior[0, 0]),
        ("logit_mu01", mu_logit_prior[0, 1]),
        ("logit_mu10", mu_logit_prior[1, 0]),
        ("logit_mu11", mu_logit_prior[1, 1]),
    ]:
        lp = lp + dist.Normal(loc, model.prior_logit_sd).log_prob(z[slices[name]][0])

    # any other logit param: Normal(0, prior_logit_sd)
    for name in [
        "logit_rho",
        "logit_eta", "logit_eta_mu", "logit_eta_cov",
    ]:
        if name in slices:
            lp = lp + dist.Normal(0.0, model.prior_logit_sd).log_prob(z[slices[name]][0])

    return lp


def theta_hat_from_unpacked(alpha, lapse_rate, sigma_init, eta_mu, eta_cov, rho, mu_init):
    theta_hat = {
        "alpha": float(alpha),
        "lapse_rate": float(lapse_rate),
        "sigma_init": float(sigma_init),
        "mu_init": [[float(mu_init[0, 0]), float(mu_init[0, 1])],
                    [float(mu_init[1, 0]), float(mu_init[1, 1])]],
        "mu_init_00": float(mu_init[0, 0]),
        "mu_init_01": float(mu_init[0, 1]),
        "mu_init_10": float(mu_init[1, 0]),
        "mu_init_11": float(mu_init[1, 1]),
    }
    if eta_mu is not None:
        theta_hat["eta_mu"] = float(eta_mu)
    if eta_cov is not None:
        theta_hat["eta_cov"] = float(eta_cov)
    if rho is not None:
        theta_hat["rho"] = float(rho)
    return theta_hat


# ---------------------------
# Fitting
# ---------------------------

def fit_point_estimate(
    model,
    data: Dict[str, Any],
    rng_seed: int = 0,
    num_steps: int = 2000,
    lr: float = 1e-2,
    map_estimate: bool = False,
    print_every: int = 200,
):
    """
    Gradient-based point estimate.
      - ML: maximize loglik
      - MAP: maximize loglik + logprior

    Returns:
      theta_hat_dict, z_hat, history_dict
    """
    rng_key = jax.random.PRNGKey(rng_seed)
    z0, slices = build_unconstrained_init(model, rng_key)

    def objective(z):
        alpha, lapse_rate, sigma_init, eta_mu, eta_cov, rho, mu_init = unpack_theta_from_z(model, z, slices)
        ll = model.log_likelihood(
            alpha, lapse_rate, data,
            eta_mu=eta_mu, eta_cov=eta_cov,
            rho=rho,
            mu_init=mu_init,
            sigma_init=sigma_init,
        )
        ll = jnp.where(jnp.isfinite(ll), ll, -1e30)
        if map_estimate:
            lp = log_prior_from_z(model, z, slices)
            lp = jnp.where(jnp.isfinite(lp), lp, -1e30)
            obj = -(ll + lp)
        else:
            obj = -ll
        return jnp.where(jnp.isfinite(obj), obj, 1e30)

    grad_fn = jax.jit(jax.value_and_grad(objective))
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    opt_state = opt.init(z0)

    z = z0
    hist_loss = []

    for t in range(1, num_steps + 1):
        loss, g = grad_fn(z)
        g = jax.tree_util.tree_map(lambda v: jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), g)
        updates, opt_state = opt.update(g, opt_state)
        z = optax.apply_updates(z, updates)

        if (t % print_every) == 0 or t == 1:
            hist_loss.append(float(loss))

    alpha, lapse_rate, sigma_init, eta_mu, eta_cov, rho, mu_init = unpack_theta_from_z(model, z, slices)
    theta_hat = theta_hat_from_unpacked(alpha, lapse_rate, sigma_init, eta_mu, eta_cov, rho, mu_init)

    history = {"loss": hist_loss}
    return theta_hat, z, history


def fit_point_estimate_multistart(
    model,
    data: Dict[str, Any],
    n_starts: int = 3,
    rng_seed: int = 0,
    num_steps: int = 2000,
    lr: float = 1e-2,
    map_estimate: bool = False,
    print_every: int = 200,
):
    """
    Run fit_point_estimate n_starts times (different random inits via rng_seed+i),
    and return the best run (lowest final objective).

    Returns:
      best_theta_hat, best_z, best_history, all_runs
    """
    runs = []
    best = None

    # JIT a final-loss function that uses theta_hat directly (no slice issues)
    def final_objective_from_theta(theta_hat):
        alpha = jnp.asarray(theta_hat["alpha"])
        lapse_rate = jnp.asarray(theta_hat["lapse_rate"])
        sigma_init = jnp.asarray(theta_hat["sigma_init"]) if "sigma_init" in theta_hat else None

        eta_mu = jnp.asarray(theta_hat["eta_mu"]) if "eta_mu" in theta_hat else None
        eta_cov = jnp.asarray(theta_hat["eta_cov"]) if "eta_cov" in theta_hat else None
        rho = jnp.asarray(theta_hat["rho"]) if "rho" in theta_hat else None
        mu_init = jnp.asarray(theta_hat["mu_init"], dtype=jnp.float32)

        ll = model.log_likelihood(
            alpha, lapse_rate, data,
            eta_mu=eta_mu, eta_cov=eta_cov,
            rho=rho,
            mu_init=mu_init,
            sigma_init=sigma_init,
        )

        if not map_estimate:
            return -ll

        # MAP: add prior (in z-space) would require z; instead compute log-prior in theta-space approximately.
        # Simpler: re-run log_prior_from_z using the run's z/slices. We do that below per-run.
        return -ll

    for i in range(n_starts):
        seed_i = rng_seed + i
        theta_hat, z_hat, hist = fit_point_estimate(
            model=model,
            data=data,
            rng_seed=seed_i,
            num_steps=num_steps,
            lr=lr,
            map_estimate=map_estimate,
            print_every=print_every,
        )

        # compute final loss properly:
        if not map_estimate:
            final_loss = float(final_objective_from_theta(theta_hat))
        else:
            # For MAP, compute using returned z_hat and that run's slices.
            z_hat = jnp.asarray(z_hat)
            _, slices = build_unconstrained_init(model, jax.random.PRNGKey(seed_i))

            alpha, lapse_rate, sigma_init, eta_mu, eta_cov, rho, mu_init = unpack_theta_from_z(model, z_hat, slices)
            ll = model.log_likelihood(
                alpha, lapse_rate, data,
                eta_mu=eta_mu, eta_cov=eta_cov,
                rho=rho,
                mu_init=mu_init,
                sigma_init=sigma_init,
            )
            lp = log_prior_from_z(model, z_hat, slices)
            final_loss = float(-(ll + lp))

        run = {
            "seed": seed_i,
            "theta_hat": theta_hat,
            "z_hat": z_hat,
            "history": hist,
            "final_loss": final_loss,
        }
        runs.append(run)

        if best is None or run["final_loss"] < best["final_loss"]:
            best = run

    return best["theta_hat"], best["z_hat"], best["history"], runs
