from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp


import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import numpy as np
from matplotlib import pyplot as plt

LOG2PI = jnp.log(2.0 * jnp.pi)


# ---------- utilities ----------

def stickbreaking_mean_pi(alpha: jnp.ndarray, K: int) -> jnp.ndarray:
    v = 1.0 / (1.0 + alpha)
    remaining = 1.0
    pis = []
    for _ in range(K - 1):
        pi_k = remaining * v
        pis.append(pi_k)
        remaining = remaining * (1.0 - v)
    pis.append(remaining)
    return jnp.stack(pis)


def _softmax_stable(logits: jnp.ndarray) -> jnp.ndarray:
    m = jnp.max(logits)
    ex = jnp.exp(logits - m)
    return ex / jnp.sum(ex)


# def _log_mvn_full(x: jnp.ndarray, mu: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
#     """
#     x: (2,)
#     mu: (K,2)
#     cov: (K,2,2)
#     returns: (K,)
#     """
#     d = x.shape[0]

#     def one(k):
#         L = jnp.linalg.cholesky(cov[k])
#         diff = x - mu[k]
#         y = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
#         maha = jnp.dot(y, y)
#         logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
#         return -0.5 * (d * LOG2PI + logdet + maha)

#     return jax.vmap(one)(jnp.arange(mu.shape[0]))
def _log_mvn_full(x: jnp.ndarray, mu: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """
    x: (2,)
    mu: (K,2)
    cov: (K,2,2)
    returns: (K,)
    """
    d = x.shape[0]

    # Final SPD guard (cheap for 2x2, batched over K)
    cov = _project_spd_2x2(cov, eps=1e-8)

    def one(k):
        L = jnp.linalg.cholesky(cov[k])
        diff = x - mu[k]
        y = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
        maha = jnp.dot(y, y)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return -0.5 * (d * LOG2PI + logdet + maha)

    return jax.vmap(one)(jnp.arange(mu.shape[0]))



# def _cov_from_stats(S, N, eps):
#     """
#     S: (..., K, 2, 2)
#     N: (..., K)
#     returns cov: (..., K, 2, 2)
#     """
#     cov = S / (N[..., :, None, None] + 1e-9)
#     cov = cov + eps * jnp.eye(2, dtype=S.dtype)
#     return cov
def _cov_from_stats(S, N, eps):
    """
    S: (..., K, 2, 2)  (scatter-like)
    N: (..., K)        (effective counts)
    eps: float         (min eigenvalue floor for SPD)
    returns cov: (..., K, 2, 2) guaranteed SPD
    """
    # Guard sufficient stats against numerical overflow/NaN before normalization.
    N = jnp.nan_to_num(N, nan=1.0, posinf=1e6, neginf=1e-6)
    N = jnp.clip(N, 1e-6, 1e6)
    S = jnp.nan_to_num(S, nan=0.0, posinf=1e6, neginf=-1e6)
    S = jnp.clip(S, -1e6, 1e6)
    cov = S / (N[..., :, None, None] + 1e-9)
    cov = _project_spd_2x2(cov, eps=eps)   # clamps eigenvalues >= eps
    return cov


def _project_spd_2x2(M, eps=1e-8):
    """
    Project (...,2,2) to SPD by symmetrizing + eigenvalue clamp.
    """
    M = 0.5 * (M + jnp.swapaxes(M, -1, -2))
    w, V = jnp.linalg.eigh(M)
    w = jnp.maximum(w, eps)
    return (V * w[..., None, :]) @ jnp.swapaxes(V, -1, -2)


# ---------- model ----------

@dataclass
class HardLabelTwoMixtureFullCov:

    config: Optional[Dict[str, Any]] = field(default=None, repr=False)

    # capacity per label
    K0: int = 3
    K1: int = 3

    # numeric stability
    eps_cov: float = 1e-3
    n_floor: float = 1e-2
    N0: float = 1.0

    # fixed init covariance scale
    sigma_init: float = 1.707
    sigma_init_min: float = 1e-3
    sigma_init_max: float = 10.0
    prior_log_sigma_init_sd: float = 0.5
    mu_min: float = 0.0
    mu_max: float = 6.0

    # prior mean for initial cluster locations by label:
    # mu_init[0, :] for label 0, mu_init[1, :] for label 1
    mu_init_prior: Tuple[Tuple[float, float], Tuple[float, float]] = ((3.0, 3.0), (3.0, 3.0))

    # forgetting on mixture counts only (correct-label only)
    use_forgetting: bool = True

    # -------- pull learning-rate options --------
    # if False -> mean uses r/N; if True -> mean uses eta_mu * r
    use_lr_mu: bool = True

    # if False -> scatter uses classic Welford increment; if True -> scaled by eta_cov * r
    use_lr_cov: bool = False

    # if True and both LR enabled -> one eta used for both mean & cov
    tie_pull_lrs: bool = True

    # priors
    prior_alpha_shape: float = 2.0
    prior_alpha_rate: float = 2.0

    prior_logit_sd: float = 1.0
    prior_lapse_logit_sd: float = 1.0

    def __post_init__(self):
        if self.config is None:
            return
        # Backward compatibility: accept and ignore deprecated push/pull config keys.
        deprecated = {"use_push_pull", "use_push_cov", "tie_push_lrs", "project_cov_spd"}
        self.config = {k: v for k, v in self.config.items() if k not in deprecated}
        unknown = [k for k in self.config.keys() if not hasattr(self, k)]
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        for k, v in self.config.items():
            setattr(self, k, v)

        # basic sanity
        if self.K0 < 1 or self.K1 < 1:
            raise ValueError("K0 and K1 must be >= 1")
        if self.mu_max <= self.mu_min:
            raise ValueError("mu_max must be > mu_min")
        if self.sigma_init_max <= self.sigma_init_min:
            raise ValueError("sigma_init_max must be > sigma_init_min")

    def _prep(self, data: Dict[str, Any]) -> Tuple[jnp.ndarray, ...]:
        tr = data["training"]
        te = data["test"]

        x_tr = jnp.asarray(tr["stimuli"], dtype=jnp.float32)
        y_tr = jnp.asarray(tr["labels"]).astype(jnp.int32)
        a_tr = jnp.asarray(tr["action"]).astype(jnp.float32)
        lost_tr = jnp.asarray(tr["trialLost"]).astype(jnp.int32)

        x_te = jnp.asarray(te["stimuli"], dtype=jnp.float32)
        y_te = jnp.asarray(te["labels"]).astype(jnp.int32)
        a_te = jnp.asarray(te["action"]).astype(jnp.float32)
        lost_te = jnp.asarray(te["trialLost"]).astype(jnp.int32)

        if x_tr.ndim != 2 or x_tr.shape[1] != 2:
            raise ValueError("training stimuli must have shape (N, 2)")
        if x_te.ndim != 2 or x_te.shape[1] != 2:
            raise ValueError("test stimuli must have shape (N, 2)")
        return x_tr, y_tr, a_tr, lost_tr, x_te, y_te, a_te, lost_te

    # ---------- core likelihood ----------

    def log_likelihood(
        self,
        alpha: jnp.ndarray,
        lapse_rate: jnp.ndarray,
        data: Dict[str, Any],
        eta_mu: jnp.ndarray | None = None,
        eta_cov: jnp.ndarray | None = None,
        rho: jnp.ndarray | None = None,
        mu_init: jnp.ndarray | None = None,
        sigma_init: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        x_tr, y_tr, a_tr, lost_tr, x_te, y_te, a_te, lost_te = self._prep(data)

        lam = jnp.clip(lapse_rate, 1e-9, 1.0 - 1e-9)
        if rho is None:
            rho = 1.0

        # pull LR defaults (when not used)
        if eta_mu is None:
            eta_mu = 1.0
        if eta_cov is None:
            eta_cov = 1.0

        if mu_init is None:
            mu_init = jnp.array(self.mu_init_prior, dtype=jnp.float32)
        else:
            mu_init = jnp.asarray(mu_init, dtype=jnp.float32)
        mu_init = jnp.clip(mu_init, self.mu_min, self.mu_max)
        if sigma_init is None:
            sigma_init = jnp.asarray(self.sigma_init, dtype=jnp.float32)
        else:
            sigma_init = jnp.asarray(sigma_init, dtype=jnp.float32)
        sigma_init = jnp.clip(sigma_init, self.sigma_init_min, self.sigma_init_max)

        # --- init label-0 mixture ---
        K0 = self.K0
        mu0 = jnp.tile(mu_init[0][None, :], (K0, 1))
        pi0_init = stickbreaking_mean_pi(alpha, K0)
        n0 = alpha * pi0_init + self.n_floor

        N0_vec = jnp.ones((K0,), dtype=jnp.float32) * self.N0
        S0 = jnp.tile(((sigma_init ** 2) * jnp.eye(2, dtype=jnp.float32))[None, :, :], (K0, 1, 1)) * N0_vec[:, None, None]

        # --- init label-1 mixture ---
        K1 = self.K1
        mu1 = jnp.tile(mu_init[1][None, :], (K1, 1))
        pi1_init = stickbreaking_mean_pi(alpha, K1)
        n1 = alpha * pi1_init + self.n_floor

        N1_vec = jnp.ones((K1,), dtype=jnp.float32) * self.N0
        S1 = jnp.tile(((sigma_init ** 2) * jnp.eye(2, dtype=jnp.float32))[None, :, :], (K1, 1, 1)) * N1_vec[:, None, None]

        def log_mix(p_model: jnp.ndarray) -> jnp.ndarray:
            p = (1.0 - lam) * p_model + lam * 0.5
            return jnp.log(jnp.clip(p, 1e-9, 1.0))

        def compute_p1_and_rtilde(mu0, S0, N0, n0, mu1, S1, N1, n1, x):
            cov0 = _cov_from_stats(S0, N0, self.eps_cov)
            cov1 = _cov_from_stats(S1, N1, self.eps_cov)

            pi0 = (n0 + 1e-12) / jnp.sum(n0 + 1e-12)
            pi1 = (n1 + 1e-12) / jnp.sum(n1 + 1e-12)

            logu0 = jnp.log(pi0 + 1e-12) + _log_mvn_full(x, mu0, cov0)  # (K0,)
            logu1 = jnp.log(pi1 + 1e-12) + _log_mvn_full(x, mu1, cov1)  # (K1,)

            finite0 = jnp.isfinite(logu0)
            finite1 = jnp.isfinite(logu1)
            any0 = jnp.any(finite0)
            any1 = jnp.any(finite1)
            logu0_safe = jnp.where(finite0, logu0, -1e30)
            logu1_safe = jnp.where(finite1, logu1, -1e30)

            logZ0 = jnp.where(any0, logsumexp(logu0_safe), -1e30)
            logZ1 = jnp.where(any1, logsumexp(logu1_safe), -1e30)

            # p1 = Z1/(Z0+Z1), robust if both evidences are degenerate
            denom = jnp.exp(logZ0 - jnp.maximum(logZ0, logZ1)) + jnp.exp(logZ1 - jnp.maximum(logZ0, logZ1))
            p1_raw = jnp.exp(logZ1 - jnp.maximum(logZ0, logZ1)) / jnp.maximum(denom, 1e-12)
            p1 = jnp.where(any0 | any1, p1_raw, 0.5)
            p1 = jnp.clip(p1, 1e-9, 1.0 - 1e-9)

            # responsibilities within each mixture (uniform fallback)
            r0_soft = jax.nn.softmax(logu0_safe)
            r1_soft = jax.nn.softmax(logu1_safe)
            r0_tilde = jnp.where(any0, r0_soft, jnp.ones_like(logu0) / logu0.shape[0])
            r1_tilde = jnp.where(any1, r1_soft, jnp.ones_like(logu1) / logu1.shape[0])

            return p1, r0_tilde, r1_tilde

        def update_gaussian(mu, S, N, r_tilde, x, *, lr_mu: float, lr_cov: float):
            
            r_tilde = jnp.nan_to_num(r_tilde, nan=0.0, posinf=0.0, neginf=0.0)
            rs = jnp.sum(r_tilde)
            r_tilde = jnp.where(rs > 0.0, r_tilde / rs, jnp.ones_like(r_tilde) / r_tilde.shape[0])

            delta = x[None, :] - mu
            delta = jnp.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            delta = jnp.clip(delta, -100.0, 100.0)

            N_new = N + r_tilde
            N_new = jnp.nan_to_num(N_new, nan=1.0, posinf=1e6, neginf=1e-6)
            N_new = jnp.clip(N_new, 1e-6, 1e6)
            if self.use_lr_mu:
                mu_new = mu + (lr_mu * r_tilde)[:, None] * delta
            else:
                mu_new = mu + (r_tilde / (N_new + 1e-12))[:, None] * delta
            mu_new = jnp.clip(mu_new, self.mu_min, self.mu_max)
            mu_new = jnp.nan_to_num(mu_new, nan=0.5 * (self.mu_min + self.mu_max), posinf=self.mu_max, neginf=self.mu_min)

            delta2 = x[None, :] - mu_new
            delta2 = jnp.nan_to_num(delta2, nan=0.0, posinf=0.0, neginf=0.0)
            delta2 = jnp.clip(delta2, -100.0, 100.0)
            dS = r_tilde[:, None, None] * (delta[:, :, None] @ delta2[:, None, :])
            dS = jnp.nan_to_num(dS, nan=0.0, posinf=1e6, neginf=-1e6)
            dS = jnp.clip(dS, -1e6, 1e6)

            if self.use_lr_cov:
                S_new = S + lr_cov * dS
            else:
                S_new = S + dS
            S_new = jnp.nan_to_num(S_new, nan=0.0, posinf=1e6, neginf=-1e6)
            S_new = jnp.clip(S_new, -1e6, 1e6)

            # ---- ROBUSTNESS: symmetrize + repair to SPD scatter ----
            # symmetrize S_new (numerical drift can break symmetry)
            S_new = 0.5 * (S_new + jnp.swapaxes(S_new, -1, -2))

            # convert to SPD covariance (eigs clamped), then map back to scatter
            cov_new = _cov_from_stats(S_new, N_new, self.eps_cov)      # SPD cov
            S_new = cov_new * (N_new[:, None, None])                   # consistent scatter

            return mu_new, S_new, N_new


        # -------- training scan --------

        def training_step(carry, inp):
            mu0, S0, N0, n0, mu1, S1, N1, n1, ll = carry
            x, y_corr, a_obs, lost = inp
            keep = 1.0 - lost.astype(jnp.float32)

            def do_nothing(args):
                return args

            def do_trial(args):
                mu0, S0, N0, n0, mu1, S1, N1, n1, ll = args

                # pre-feedback inference
                p1, r0_tilde, r1_tilde = compute_p1_and_rtilde(mu0, S0, N0, n0, mu1, S1, N1, n1, x)
                p_model = jnp.where(a_obs > 0.5, p1, 1.0 - p1)
                ll = ll + log_mix(p_model)

                # choose pull learning rates (shared vs separate)
                pull_mu = eta_mu
                pull_cov = eta_cov

                def upd_correct0(st):
                    mu0, S0, N0, n0, mu1, S1, N1, n1, ll = st

                    # forgetting affects only correct-label counts
                    if self.use_forgetting:
                        n0 = rho * n0
                    n0_new = n0 + r0_tilde

                    mu0_new, S0_new, N0_new = update_gaussian(
                        mu0, S0, N0, r0_tilde, x,
                        lr_mu=pull_mu, lr_cov=pull_cov
                    )
                    return (mu0_new, S0_new, N0_new, n0_new, mu1, S1, N1, n1, ll)

                def upd_correct1(st):
                    mu0, S0, N0, n0, mu1, S1, N1, n1, ll = st

                    if self.use_forgetting:
                        n1 = rho * n1
                    n1_new = n1 + r1_tilde

                    mu1_new, S1_new, N1_new = update_gaussian(
                        mu1, S1, N1, r1_tilde, x,
                        lr_mu=pull_mu, lr_cov=pull_cov
                    )
                    return (mu0, S0, N0, n0, mu1_new, S1_new, N1_new, n1_new, ll)

                return lax.cond(y_corr == 0, upd_correct0, upd_correct1,
                                (mu0, S0, N0, n0, mu1, S1, N1, n1, ll))

            out = lax.cond(keep > 0.5, do_trial, do_nothing,
                           (mu0, S0, N0, n0, mu1, S1, N1, n1, ll))
            return out, None

        # -------- test scan (no learning) --------

        def test_step(carry, inp):
            mu0, S0, N0, n0, mu1, S1, N1, n1, ll = carry
            x, a_obs, lost = inp
            keep = 1.0 - lost.astype(jnp.float32)

            def do_nothing(args):
                return args

            def do_score(args):
                mu0, S0, N0, n0, mu1, S1, N1, n1, ll = args
                p1, _, _ = compute_p1_and_rtilde(mu0, S0, N0, n0, mu1, S1, N1, n1, x)
                p_model = jnp.where(a_obs > 0.5, p1, 1.0 - p1)
                ll = ll + log_mix(p_model)
                return (mu0, S0, N0, n0, mu1, S1, N1, n1, ll)

            out = lax.cond(keep > 0.5, do_score, do_nothing,
                           (mu0, S0, N0, n0, mu1, S1, N1, n1, ll))
            return out, None

        init = (mu0, S0, N0_vec, n0, mu1, S1, N1_vec, n1, 0.0)
        (mu0, S0, N0_vec, n0, mu1, S1, N1_vec, n1, ll), _ = lax.scan(
            training_step, init, (x_tr, y_tr, a_tr, lost_tr)
        )
        (mu0, S0, N0_vec, n0, mu1, S1, N1_vec, n1, ll), _ = lax.scan(
            test_step, (mu0, S0, N0_vec, n0, mu1, S1, N1_vec, n1, ll), (x_te, a_te, lost_te)
        )
        return ll

    # ---------- numpyro model ----------

    def numpyro_model(self, data: Dict[str, Any]):
        alpha = numpyro.sample("alpha", dist.Gamma(self.prior_alpha_shape, self.prior_alpha_rate))

        logit_lapse = numpyro.sample("logit_lapse", dist.Normal(0.0, self.prior_lapse_logit_sd))
        lapse_rate = jax.nn.sigmoid(logit_lapse)

        # forgetting (optional)
        rho = None
        if self.use_forgetting:
            logit_rho = numpyro.sample("logit_rho", dist.Normal(0.0, self.prior_logit_sd))
            rho = jax.nn.sigmoid(logit_rho)

        mu_prior = jnp.asarray(self.mu_init_prior, dtype=jnp.float32)
        mu_unit = (mu_prior - self.mu_min) / (self.mu_max - self.mu_min)
        mu_unit = jnp.clip(mu_unit, 1e-6, 1.0 - 1e-6)
        logit_mu_prior = jnp.log(mu_unit) - jnp.log1p(-mu_unit)

        logit_mu_init = numpyro.sample(
            "logit_mu_init",
            dist.Normal(logit_mu_prior, self.prior_logit_sd).to_event(2),
        )
        mu_init = self.mu_min + (self.mu_max - self.mu_min) * jax.nn.sigmoid(logit_mu_init)

        log_sigma_init = numpyro.sample(
            "log_sigma_init",
            dist.Normal(jnp.log(self.sigma_init), self.prior_log_sigma_init_sd),
        )
        sigma_init = jnp.exp(log_sigma_init)
        sigma_init = jnp.clip(sigma_init, self.sigma_init_min, self.sigma_init_max)

        # pull learning rates
        eta_mu = None
        eta_cov = None

        if self.use_lr_mu or self.use_lr_cov:
            if self.tie_pull_lrs and self.use_lr_mu and self.use_lr_cov:
                logit_eta = numpyro.sample("logit_eta", dist.Normal(0.0, self.prior_logit_sd))
                eta = jax.nn.sigmoid(logit_eta)
                eta_mu = eta
                eta_cov = eta
            else:
                if self.use_lr_mu:
                    logit_eta_mu = numpyro.sample("logit_eta_mu", dist.Normal(0.0, self.prior_logit_sd))
                    eta_mu = jax.nn.sigmoid(logit_eta_mu)
                if self.use_lr_cov:
                    logit_eta_cov = numpyro.sample("logit_eta_cov", dist.Normal(0.0, self.prior_logit_sd))
                    eta_cov = jax.nn.sigmoid(logit_eta_cov)

        ll = self.log_likelihood(
            alpha, lapse_rate, data,
            eta_mu=eta_mu, eta_cov=eta_cov,
            rho=rho,
            mu_init=mu_init,
            sigma_init=sigma_init,
        )
        numpyro.factor("ll", ll)

    def fit_nuts(
        self,
        data: Dict[str, Any],
        rng_seed: int = 0,
        num_warmup: int = 1500,
        num_samples: int = 1500,
        num_chains: int = 2,
        chain_method: str = "parallel",
        progress_bar: bool = True,
    ) -> MCMC:
        rng_key = jax.random.PRNGKey(rng_seed)
        kernel = NUTS(self.numpyro_model, target_accept_prob=0.95)
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
        )
        mcmc.run(rng_key, data=data)
        return mcmc

    # ---------- tracing (kept consistent with likelihood) ----------

    def trace_sufficient_stats(
        self,
        alpha: jnp.ndarray,
        lapse_rate: jnp.ndarray,
        data: dict,
        eta_mu: jnp.ndarray | None = None,
        eta_cov: jnp.ndarray | None = None,
        rho: jnp.ndarray | None = None,
        mu_init: jnp.ndarray | None = None,
        sigma_init: jnp.ndarray | None = None,
        rng_key: jax.random.PRNGKey | None = None,
        include_initial: bool = True,
        return_cov: bool = True,
        return_pi: bool = True,
        return_rtilde: bool = False,
        return_behavior: bool = True,
    ):
        x_tr, y_tr, a_tr, lost_tr, x_te, y_te, a_te, lost_te = self._prep(data)

        lam = jnp.clip(lapse_rate, 1e-9, 1.0 - 1e-9)

        if rho is None:
            rho = 1.0
        if eta_mu is None:
            eta_mu = 1.0
        if eta_cov is None:
            eta_cov = 1.0
        if mu_init is None:
            mu_init = jnp.array(self.mu_init_prior, dtype=jnp.float32)
        else:
            mu_init = jnp.asarray(mu_init, dtype=jnp.float32)
        mu_init = jnp.clip(mu_init, self.mu_min, self.mu_max)
        if sigma_init is None:
            sigma_init = jnp.asarray(self.sigma_init, dtype=jnp.float32)
        else:
            sigma_init = jnp.asarray(sigma_init, dtype=jnp.float32)
        sigma_init = jnp.clip(sigma_init, self.sigma_init_min, self.sigma_init_max)

        K0, K1 = self.K0, self.K1

        # init label-0
        mu0_0 = jnp.tile(mu_init[0][None, :], (K0, 1))
        pi0_init = stickbreaking_mean_pi(alpha, K0)
        n0_0 = alpha * pi0_init + self.n_floor
        N0_0 = jnp.ones((K0,), dtype=jnp.float32) * self.N0
        S0_0 = jnp.tile(((sigma_init ** 2) * jnp.eye(2, dtype=jnp.float32))[None, :, :], (K0, 1, 1)) * N0_0[:, None, None]

        # init label-1
        mu1_0 = jnp.tile(mu_init[1][None, :], (K1, 1))
        pi1_init = stickbreaking_mean_pi(alpha, K1)
        n1_0 = alpha * pi1_init + self.n_floor
        N1_0 = jnp.ones((K1,), dtype=jnp.float32) * self.N0
        S1_0 = jnp.tile(((sigma_init ** 2) * jnp.eye(2, dtype=jnp.float32))[None, :, :], (K1, 1, 1)) * N1_0[:, None, None]

        # concat phases
        x_all = jnp.concatenate([x_tr, x_te], axis=0)
        y_all = jnp.concatenate([y_tr, y_te], axis=0)
        a_all = jnp.concatenate([a_tr, a_te], axis=0)
        lost_all = jnp.concatenate([lost_tr, lost_te], axis=0)
        is_train = jnp.concatenate(
            [jnp.ones((x_tr.shape[0],), dtype=jnp.int32),
             jnp.zeros((x_te.shape[0],), dtype=jnp.int32)],
            axis=0
        )
        phase_idx = x_tr.shape[0]
        T = x_all.shape[0]

        def compute_p1_and_rtilde(mu0, S0, N0, n0, mu1, S1, N1, n1, x):
            cov0 = _cov_from_stats(S0, N0, self.eps_cov)
            cov1 = _cov_from_stats(S1, N1, self.eps_cov)

            pi0 = (n0 + 1e-12) / jnp.sum(n0 + 1e-12)
            pi1 = (n1 + 1e-12) / jnp.sum(n1 + 1e-12)

            logu0 = jnp.log(pi0 + 1e-12) + _log_mvn_full(x, mu0, cov0)  # (K0,)
            logu1 = jnp.log(pi1 + 1e-12) + _log_mvn_full(x, mu1, cov1)  # (K1,)

            finite0 = jnp.isfinite(logu0)
            finite1 = jnp.isfinite(logu1)
            any0 = jnp.any(finite0)
            any1 = jnp.any(finite1)
            logu0_safe = jnp.where(finite0, logu0, -1e30)
            logu1_safe = jnp.where(finite1, logu1, -1e30)

            logZ0 = jnp.where(any0, logsumexp(logu0_safe), -1e30)
            logZ1 = jnp.where(any1, logsumexp(logu1_safe), -1e30)

            denom = jnp.exp(logZ0 - jnp.maximum(logZ0, logZ1)) + jnp.exp(logZ1 - jnp.maximum(logZ0, logZ1))
            p1_raw = jnp.exp(logZ1 - jnp.maximum(logZ0, logZ1)) / jnp.maximum(denom, 1e-12)
            p1 = jnp.where(any0 | any1, p1_raw, 0.5)
            p1 = jnp.clip(p1, 1e-9, 1.0 - 1e-9)

            r0_soft = jax.nn.softmax(logu0_safe)
            r1_soft = jax.nn.softmax(logu1_safe)
            r0_tilde = jnp.where(any0, r0_soft, jnp.ones_like(logu0) / logu0.shape[0])
            r1_tilde = jnp.where(any1, r1_soft, jnp.ones_like(logu1) / logu1.shape[0])

            return p1, r0_tilde, r1_tilde

        def update_gaussian(mu, S, N, r_tilde, x, *, lr_mu: float, lr_cov: float):
            
            r_tilde = jnp.nan_to_num(r_tilde, nan=0.0, posinf=0.0, neginf=0.0)
            rs = jnp.sum(r_tilde)
            r_tilde = jnp.where(rs > 0.0, r_tilde / rs, jnp.ones_like(r_tilde) / r_tilde.shape[0])

            delta = x[None, :] - mu
            delta = jnp.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            delta = jnp.clip(delta, -100.0, 100.0)

            N_new = N + r_tilde
            N_new = jnp.nan_to_num(N_new, nan=1.0, posinf=1e6, neginf=1e-6)
            N_new = jnp.clip(N_new, 1e-6, 1e6)
            if self.use_lr_mu:
                mu_new = mu + (lr_mu * r_tilde)[:, None] * delta
            else:
                mu_new = mu + (r_tilde / (N_new + 1e-12))[:, None] * delta
            mu_new = jnp.clip(mu_new, self.mu_min, self.mu_max)
            mu_new = jnp.nan_to_num(mu_new, nan=0.5 * (self.mu_min + self.mu_max), posinf=self.mu_max, neginf=self.mu_min)

            delta2 = x[None, :] - mu_new
            delta2 = jnp.nan_to_num(delta2, nan=0.0, posinf=0.0, neginf=0.0)
            delta2 = jnp.clip(delta2, -100.0, 100.0)
            dS = r_tilde[:, None, None] * (delta[:, :, None] @ delta2[:, None, :])
            dS = jnp.nan_to_num(dS, nan=0.0, posinf=1e6, neginf=-1e6)
            dS = jnp.clip(dS, -1e6, 1e6)

            if self.use_lr_cov:
                S_new = S + lr_cov * dS
            else:
                S_new = S + dS
            S_new = jnp.nan_to_num(S_new, nan=0.0, posinf=1e6, neginf=-1e6)
            S_new = jnp.clip(S_new, -1e6, 1e6)

            # ---- ROBUSTNESS: symmetrize + repair to SPD scatter ----
            # symmetrize S_new (numerical drift can break symmetry)
            S_new = 0.5 * (S_new + jnp.swapaxes(S_new, -1, -2))

            # convert to SPD covariance (eigs clamped), then map back to scatter
            cov_new = _cov_from_stats(S_new, N_new, self.eps_cov)      # SPD cov
            S_new = cov_new * (N_new[:, None, None])                   # consistent scatter

            return mu_new, S_new, N_new


        nan = jnp.array(jnp.nan, dtype=jnp.float32)
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
            do_sample = False
        else:
            do_sample = True

        def step(carry, inp):
            (mu0, S0, N0, n0, mu1, S1, N1, n1, key) = carry
            x, y_corr, a_obs, lost, train_flag = inp
            keep = 1.0 - lost.astype(jnp.float32)

            # defaults for lost trials
            p1_out = nan
            p_act1_out = nan
            exp_acc_out = nan
            samp_acc_out = nan
            samp_a_out = nan
            r0_out = jnp.ones((K0,), dtype=jnp.float32) / K0
            r1_out = jnp.ones((K1,), dtype=jnp.float32) / K1

            def do_lost(args):
                mu0, S0, N0, n0, mu1, S1, N1, n1, key = args
                return (mu0, S0, N0, n0, mu1, S1, N1, n1, key), (
                    mu0, S0, N0, n0, mu1, S1, N1, n1,
                    p1_out, p_act1_out, exp_acc_out, samp_acc_out, samp_a_out,
                    r0_out, r1_out
                )

            def do_trial(args):
                mu0, S0, N0, n0, mu1, S1, N1, n1, key = args

                p1, r0_tilde, r1_tilde = compute_p1_and_rtilde(mu0, S0, N0, n0, mu1, S1, N1, n1, x)

                p_act1 = (1.0 - lam) * p1 + lam * 0.5
                p_act1 = jnp.clip(p_act1, 1e-9, 1.0 - 1e-9)

                y_f = y_corr.astype(jnp.float32)
                exp_acc = y_f * p_act1 + (1.0 - y_f) * (1.0 - p_act1)

                key, subk = jax.random.split(key)
                a_samp = jnp.where(do_sample, jax.random.bernoulli(subk, p_act1).astype(jnp.float32), nan)
                samp_acc = jnp.where(do_sample, (a_samp == y_f).astype(jnp.float32), nan)

                def do_test_update(st):
                    return st

                def do_train_update(st):
                    mu0, S0, N0, n0, mu1, S1, N1, n1 = st

                    pull_mu = eta_mu
                    pull_cov = eta_cov
                    def upd0(st2):
                        mu0, S0, N0, n0, mu1, S1, N1, n1 = st2
                        if self.use_forgetting:
                            n0 = rho * n0
                        n0_new = n0 + r0_tilde
                        mu0_new, S0_new, N0_new = update_gaussian(mu0, S0, N0, r0_tilde, x, lr_mu=pull_mu, lr_cov=pull_cov)
                        return (mu0_new, S0_new, N0_new, n0_new, mu1, S1, N1, n1)

                    def upd1(st2):
                        mu0, S0, N0, n0, mu1, S1, N1, n1 = st2
                        if self.use_forgetting:
                            n1 = rho * n1
                        n1_new = n1 + r1_tilde
                        mu1_new, S1_new, N1_new = update_gaussian(mu1, S1, N1, r1_tilde, x, lr_mu=pull_mu, lr_cov=pull_cov)
                        return (mu0, S0, N0, n0, mu1_new, S1_new, N1_new, n1_new)

                    return lax.cond(y_corr == 0, upd0, upd1, (mu0, S0, N0, n0, mu1, S1, N1, n1))

                mu0n, S0n, N0n, n0n, mu1n, S1n, N1n, n1n = lax.cond(
                    train_flag == 1,
                    do_train_update,
                    do_test_update,
                    (mu0, S0, N0, n0, mu1, S1, N1, n1),
                )

                return (mu0n, S0n, N0n, n0n, mu1n, S1n, N1n, n1n, key), (
                    mu0n, S0n, N0n, n0n, mu1n, S1n, N1n, n1n,
                    p1, p_act1, exp_acc, samp_acc, a_samp,
                    r0_tilde, r1_tilde
                )

            return lax.cond(keep > 0.5, do_trial, do_lost,
                            (mu0, S0, N0, n0, mu1, S1, N1, n1, key))

        init = (mu0_0, S0_0, N0_0, n0_0, mu1_0, S1_0, N1_0, n1_0, rng_key)
        _, outs = lax.scan(step, init, (x_all, y_all, a_all, lost_all, is_train))

        (mu0_hist, S0_hist, N0_hist, n0_hist,
         mu1_hist, S1_hist, N1_hist, n1_hist,
         p1_hist, p_act1_hist, exp_acc_hist, samp_acc_hist, samp_a_hist,
         r0_hist, r1_hist) = outs

        if include_initial:
            mu0_hist = jnp.concatenate([mu0_0[None, ...], mu0_hist], axis=0)
            S0_hist  = jnp.concatenate([S0_0[None, ...],  S0_hist], axis=0)
            N0_hist  = jnp.concatenate([N0_0[None, ...],  N0_hist], axis=0)
            n0_hist  = jnp.concatenate([n0_0[None, ...],  n0_hist], axis=0)

            mu1_hist = jnp.concatenate([mu1_0[None, ...], mu1_hist], axis=0)
            S1_hist  = jnp.concatenate([S1_0[None, ...],  S1_hist], axis=0)
            N1_hist  = jnp.concatenate([N1_0[None, ...],  N1_hist], axis=0)
            n1_hist  = jnp.concatenate([n1_0[None, ...],  n1_hist], axis=0)

        out = {
            "mu0": mu0_hist, "S0": S0_hist, "N0": N0_hist, "n0": n0_hist,
            "mu1": mu1_hist, "S1": S1_hist, "N1": N1_hist, "n1": n1_hist,
            "phase_idx": phase_idx,
        }

        if return_cov:
            out["cov0"] = _cov_from_stats(S0_hist, N0_hist, self.eps_cov)
            out["cov1"] = _cov_from_stats(S1_hist, N1_hist, self.eps_cov)

        if return_pi:
            out["pi0"] = (n0_hist + 1e-12) / jnp.sum(n0_hist + 1e-12, axis=1, keepdims=True)
            out["pi1"] = (n1_hist + 1e-12) / jnp.sum(n1_hist + 1e-12, axis=1, keepdims=True)

        if return_behavior:
            def split(arr):
                return arr[:phase_idx], arr[phase_idx:]

            p1_tr, p1_te = split(p1_hist)
            p_act1_tr, p_act1_te = split(p_act1_hist)
            exp_acc_tr, exp_acc_te = split(exp_acc_hist)
            samp_acc_tr, samp_acc_te = split(samp_acc_hist)
            samp_a_tr, samp_a_te = split(samp_a_hist)

            out["training"] = {"p1": p1_tr, "p_act1": p_act1_tr, "expected_acc": exp_acc_tr, "sampled_acc": samp_acc_tr}
            out["test"]     = {"p1": p1_te, "p_act1": p_act1_te, "expected_acc": exp_acc_te, "sampled_acc": samp_acc_te}
            if do_sample:
                out["training"]["sampled_action"] = samp_a_tr
                out["test"]["sampled_action"] = samp_a_te

        if return_rtilde:
            out["r0_tilde"] = r0_hist
            out["r1_tilde"] = r1_hist

        return out

    # ---------- plotting helpers (unchanged API) ----------

    def plot_pi_over_time(self, params: dict, data: dict, figsize=(10, 4)):
        traced = self.trace_sufficient_stats(
            alpha=jnp.asarray(params["alpha"]),
            lapse_rate=jnp.asarray(params.get("lapse_rate", 0.0)),
            data=data,
            eta_mu=jnp.asarray(params.get("eta_mu", 1.0)),
            eta_cov=jnp.asarray(params.get("eta_cov", 1.0)),
            rho=jnp.asarray(params.get("rho", 1.0)),
            mu_init=jnp.asarray(params.get("mu_init", self.mu_init_prior)),
            sigma_init=jnp.asarray(params.get("sigma_init", self.sigma_init)),
            include_initial=True,
            return_pi=True,
            return_cov=False,
            return_rtilde=False,
            return_behavior=False,
        )
        pi0 = np.array(traced["pi0"])
        pi1 = np.array(traced["pi1"])
        phase_idx = int(traced["phase_idx"])
        x = np.arange(pi0.shape[0])

        plt.figure(figsize=figsize)
        for k in range(pi0.shape[1]):
            plt.plot(x, pi0[:, k], label=f"$\\pi^0_{k+1}$")
        for k in range(pi1.shape[1]):
            plt.plot(x, pi1[:, k], linestyle="--", label=f"$\\pi^1_{k+1}$")

        plt.axvline(phase_idx, linestyle=":")
        plt.xlabel("Trial (0 = initial)")
        plt.ylabel("Mixture weights")
        plt.title("Mixture weights over time (label 0 solid, label 1 dashed)")
        plt.ylim(-0.02, 1.02)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.show()

    def plot_p1_over_time(self, params: dict, data: dict, figsize=(10, 3)):
        traced = self.trace_sufficient_stats(
            alpha=jnp.asarray(params["alpha"]),
            lapse_rate=jnp.asarray(params.get("lapse_rate", 0.0)),
            data=data,
            eta_mu=jnp.asarray(params.get("eta_mu", 1.0)),
            eta_cov=jnp.asarray(params.get("eta_cov", 1.0)),
            rho=jnp.asarray(params.get("rho", 1.0)),
            mu_init=jnp.asarray(params.get("mu_init", self.mu_init_prior)),
            sigma_init=jnp.asarray(params.get("sigma_init", self.sigma_init)),
            include_initial=True,
            return_pi=False,
            return_cov=False,
            return_rtilde=False,
            return_behavior=True,
        )
        p1_tr = np.array(traced["training"]["p1"])
        p1_te = np.array(traced["test"]["p1"])
        p1 = np.concatenate([p1_tr, p1_te], axis=0)
        phase_idx = int(traced["phase_idx"])

        x = np.arange(p1.shape[0])
        plt.figure(figsize=figsize)
        plt.plot(x, p1)
        plt.axvline(phase_idx, linestyle=":")
        plt.xlabel("Trial")
        plt.ylabel("$p_1(x_t)$")
        plt.title("Predicted probability of label 1 over time (pre-feedback)")
        plt.ylim(-0.02, 1.02)
        plt.tight_layout()
        plt.show()
