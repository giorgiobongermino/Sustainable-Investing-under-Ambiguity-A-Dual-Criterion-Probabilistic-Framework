# ===========================
# Mean vector from user logp on the simplex via NUTS (NumPyro)
# ===========================

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # prima di importare jax

import jax, jax.numpy as jnp
from typing import Callable
import jax
import jax.numpy as jnp
from jax import random, vmap
from numpyro.infer import NUTS, MCMC
from numpyro.distributions.transforms import StickBreakingTransform

from typing import Callable, Optional, Literal
import jax
import jax.numpy as jnp
from jax import random, vmap
from numpyro.infer import NUTS, MCMC
from numpyro.distributions.transforms import StickBreakingTransform
import numpy as np
from scipy.optimize import minimize
def _z_for_uniform(K: int, dtype=jnp.float64) -> jnp.ndarray:
    """
    z che mappa via stick-breaking in p=(1/K,...,1/K).
    Formula: z_i = -log(K - i), i=1..K-1
    """
    return -jnp.log(jnp.arange(K - 1, 0, -1, dtype=dtype))  # [-log(K-1), ..., -log(1)=0]

def mean_simplex_from_logp(
    logp_user: Callable[[jnp.ndarray], jnp.ndarray],
    K: int,
    *,
    num_warmup: int = 2000,
    num_samples: int = 4000,
    num_chains: int = 4,
    chain_method: str = "vectorized",       # "vectorized" | "parallel" | "sequential"
    target_accept_prob: float = 0.9,
    rng_seed: int = 0,
    init_scale: float = 1,
    dense_mass: bool = True,                # su K grandi spesso aiuta
    init_mode: Literal["uniform", "random", "custom"] = "uniform",
    init_jitter: float = 1.0,
    init_custom_z: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Esegue NUTS su p ∈ Δ^K (simplesso) usando la tua log-densità logp_user(p)
    e restituisce il vettore delle medie E[p].

    Parametri chiave per l'inizializzazione:
      - init_mode="uniform": parte da p0=(1/K,...,1/K)
      - init_mode="random":  z ~ N(0, init_scale^2)
      - init_mode="custom":  usa init_custom_z come z0 (shape (K-1,) o (num_chains,K-1))
      - init_jitter: aggiunge rumore a z0 (utile se logp usa log(p))

    Ritorna:
      - means: jnp.ndarray (K,) con la media per componente.
    """
    t = StickBreakingTransform()

    # Stick-breaking + Jacobiano con supporto batch
    def _to_p(z):
        return t(z) if z.ndim == 1 else vmap(t)(z)

    def _ldj(z, p):
        fn = t.log_abs_det_jacobian
        return fn(z, p) if z.ndim == 1 else vmap(fn)(z, p)

    # Adatta automaticamente logp_user a batch se serve
    def _logp(p):
        y = logp_user(p)
        if p.ndim > 1 and y.ndim == 0:
            y = vmap(logp_user)(p)
        return y

    # potential_fn nello spazio non vincolato (z ∈ R^{K-1})
    def potential_fn(params):
        z = params["z"]
        p = _to_p(z)
        ldj = _ldj(z, p)
        return -(_logp(p) + ldj)

    kernel = NUTS(potential_fn=potential_fn, target_accept_prob=target_accept_prob, dense_mass=dense_mass)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=True,
    )

    key = random.PRNGKey(rng_seed)

    # -------- init_params con modalità richieste --------
    if init_mode == "uniform":
        z0 = _z_for_uniform(K, dtype=jnp.float64)
        if chain_method == "vectorized":
            z0 = jnp.broadcast_to(z0, (num_chains, K - 1))
        if init_jitter > 0.0:
            noise = init_jitter * random.normal(key, z0.shape)
            z0 = z0 + noise
        init_params = {"z": z0}
    elif init_mode == "random":
        if chain_method == "vectorized":
            init_params = {"z": init_scale * random.normal(key, (num_chains, K - 1))}
        else:
            init_params = {"z": init_scale * random.normal(key, (K - 1,))}
    elif init_mode == "custom":
        if init_custom_z is None:
            raise ValueError("init_mode='custom' richiede init_custom_z.")
        z0 = jnp.asarray(init_custom_z)
        # Consenti shape (K-1,) o (num_chains, K-1)
        if chain_method == "vectorized":
            if z0.ndim == 1:
                z0 = jnp.broadcast_to(z0, (num_chains, K - 1))
            elif z0.shape != (num_chains, K - 1):
                raise ValueError(f"init_custom_z deve avere shape {(num_chains, K-1)} o {(K-1,)}.")
        else:
            if z0.shape != (K - 1,):
                raise ValueError(f"init_custom_z deve avere shape {(K-1,)}.")
        if init_jitter > 0.0:
            z0 = z0 + init_jitter * random.normal(key, z0.shape)
        init_params = {"z": z0}
    else:
        raise ValueError("init_mode deve essere 'uniform', 'random' o 'custom'.")

    # -------- run MCMC --------
    mcmc.run(key, init_params=init_params)

    # Campioni su z → p, poi media per componente
    z = mcmc.get_samples(group_by_chain=False)["z"]  # (draws, K-1)
    p = vmap(t)(z)                                   # (draws, K)
    means = jnp.mean(p, axis=0)                      # (K,)
    return means

def expectation_DPU(k,m,logp1,logp2,dim):
    acceptance_rate_target=0.95
    N_samples=8000
    mean_1=mean_simplex_from_logp(logp1,dim,target_accept_prob=acceptance_rate_target,num_samples=N_samples)
    mean_2=mean_simplex_from_logp(logp2,dim,target_accept_prob=acceptance_rate_target,num_samples=N_samples)
    lamba_esg_hat=(k*m+1)/(k+1)
    return (1-lamba_esg_hat)*mean_1+lamba_esg_hat*mean_2

def simplex_optimize(fun, J: int, tol: float = 1e-6, eps: float = 0.0):
    """Massimizza fun su {w_i ≥ −ε, Σ w_i = 1}."""
    bounds = [(-eps, 1.0 + eps)] * J
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    x0 = np.full(J, 1.0 / J)
    res = minimize(lambda w: -fun(w), x0,
                   method='SLSQP', bounds=bounds, constraints=cons,
                   options={'ftol': tol, 'disp': False})
    return res.x, -res.fun, res
def mean_variance_objective(x, mu, sigma, lmbd):
    return -(lmbd*np.dot(x, mu) - (1-lmbd) * np.dot(x, np.dot(sigma, x)))
def double_mean_variance_objective(x,mu_f,mu_e,sigma_f,sigma_e,lmbd_f,lmbd_e,esg_preference,esg_trust):
    mean_variance_f=-mean_variance_objective(x,mu_f,sigma_f,lmbd_f)
    mean_variance_e=-mean_variance_objective(x,mu_e,sigma_e,lmbd_e)
    return ((1-esg_preference*(esg_trust/(esg_trust+1)))*mean_variance_f+esg_preference*(esg_trust/(esg_trust+1))*mean_variance_e)
