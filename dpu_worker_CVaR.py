import os

# === Added: empirical CVaR helper (stable, vectorized) ===
def _empirical_cvar_from_losses(L, alpha: float):
    import numpy as _np
    L = _np.asarray(L, float)
    if L.size == 0 or not _np.isfinite(L).any():
        return _np.nan
    L = L[_np.isfinite(L)]
    s = _np.sort(L)
    T = s.size
    k = int(_np.ceil(alpha * T))
    k = max(1, min(k, T))
    var_alpha = s[k-1]
    tail_sum  = s[k:].sum()
    frac      = k - alpha * T
    return float((frac * var_alpha + tail_sum) / ((1.0 - alpha) * T))


def run_one_dpu_job(cfg):
    """
    MEU: Fin = mean–CVaR empirico (NumPy), ESG = mean–variance (NumPy)  -> no JAX/tracer
    DPU: Fin = mean–CVaR empirico (JAX),    ESG = mean–variance (JAX)
    """
    try:
        # ====== ENV prima di import JAX/NumPyro ======
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
        os.environ["OMP_NUM_THREADS"]      = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"]      = "1"
        os.environ["NUMEXPR_NUM_THREADS"]  = "1"

        # ====== import ======
        import random as _py_random
        import numpy as np
        import jax, jax.numpy as jnp
        from jax import lax
        from numpyro import set_platform, enable_x64
        set_platform("cpu"); enable_x64()

        from codice_base import expectation_DPU, simplex_optimize

        import numpy as _np
        import pandas as pd

        # ---------- helper: stacca eventuali jax.Array/Tracer in NumPy ----------
        def _to_host_np(x, dtype=float):
            try:
                x = jax.device_get(x)
            except Exception:
                pass
            return np.asarray(x, dtype=dtype).reshape(-1)

        # ====== cfg ======
        m        = float(cfg["m"])
        lfin     = float(cfg["lmbd_fin"])
        lesg     = float(cfg["lmbd_esg"])
        k_val    = float(cfg["k"])
        n_assets = int(cfg["n_assets"])
        rng_seed = int(cfg.get("rng_seed", 0))
        alpha_fin = float(cfg.get("alpha_fin", 0.95))

        _py_random.seed(rng_seed)
        np.random.seed(rng_seed % (2**32 - 1))

        mu_esg_np    = _np.asarray(cfg["mu_esg_np"], dtype=float)
        Sigma_esg_np = _np.asarray(cfg["Sigma_esg_np"], dtype=float)

        nu_fin = float(cfg.get("nu_fin", 1.0))
        nu_esg = float(cfg.get("nu_esg", 1.0))

        # ====== dati in-sample per MEU (DataFrame rendimenti % settimanali) ======
        fin_df = cfg.get("fin_df", None)
        if fin_df is None or not isinstance(fin_df, pd.DataFrame):
            raise ValueError("Per il mean–CVaR empirico finanziario è necessario 'fin_df' (pandas.DataFrame) in cfg.")
        if fin_df.shape[1] < n_assets:
            raise ValueError(f"fin_df ha {fin_df.shape[1]} colonne < n_assets={n_assets}")
        fin_df = fin_df.iloc[:, :n_assets].copy()

        # ====== dataset numerici per componente JAX (DPU) ======
        fin_np_vals  = cfg.get("fin_np", None)      # T x N in %
        fin_mask_vals = cfg.get("fin_mask", None)   # T x N bool
        if fin_np_vals is None or fin_mask_vals is None:
            raise ValueError("La cfg deve contenere 'fin_np' (float[T,N]) e 'fin_mask' (bool[T,N]) per la parte JAX.")
        fin_np_vals = _np.array(fin_np_vals, dtype=float, copy=True)
        fin_mask_vals = _np.array(fin_mask_vals, dtype=bool, copy=False)
        fin_np_vals[~fin_mask_vals] = 0.0  # ripulisci i NaN per JAX

        # ====== PRECOMP: R e M per MEU (NumPy, vettoriale) ======
        R_np = fin_df.to_numpy(dtype=float, copy=False) / 100.0  # frazione
        M_np = np.isfinite(R_np).astype(float)
        R_np = np.nan_to_num(R_np, nan=0.0)  # T x N

        # ====== PRECOMP: tensori JAX per DPU ======
        R_percent_j = jnp.asarray(fin_np_vals)   # T x N in %
        M_bool_j    = jnp.asarray(fin_mask_vals) # T x N bool

        # ========= MEU (tutto NumPy) =========
        def portfolio_series_np(w):
            w = _to_host_np(w)
            num = R_np @ w                 # (T,)
            den = M_np @ w                 # (T,)
            den = np.where(np.abs(den) > 1e-15, den, np.nan)
            return num / den               # frazione

        def fin_mean_cvar_emp_utility_np(x, lmbd, alpha):
            r = portfolio_series_np(x)
            if r.size == 0 or np.all(np.isnan(r)):
                return -1e300
            mean_r = float(np.nanmean(r))
            L = -r
            cvar = _empirical_cvar_from_losses(L, alpha)
            if not np.isfinite(cvar):
                return -1e300

            return float(lmbd * mean_r - (1.0 - lmbd) * cvar)*100

        def esg_mean_variance_utility_np(x, mu, Sigma, lmbd):
            x = _to_host_np(x)
            
            return float(lmbd * x.dot(mu) - (1.0 - lmbd) * x.dot(Sigma @ x))

        def MEU_overall_np(x):
            u_fin = fin_mean_cvar_emp_utility_np(x, lfin, alpha_fin)
            u_esg = esg_mean_variance_utility_np(x, mu_esg_np, Sigma_esg_np, lesg)
            return float((1.0 - m * k_val) * u_fin + (m * k_val) * u_esg)

        w_MEU, _, _ = simplex_optimize(MEU_overall_np, n_assets)
        w_MEU = _to_host_np(w_MEU)
        if w_MEU.sum() != 0:
            w_MEU = w_MEU / w_MEU.sum()

        hhi_meu = float(np.sum(w_MEU**2))

        # ========= DPU (tutto JAX) =========
        def portfolio_series_jax(x):
            # x: jnp[N]
            num = (R_percent_j / 100.0) @ x                 # (T,)
            den = (M_bool_j.astype(x.dtype)) @ x            # (T,)
            den = jnp.where(jnp.abs(den) > 1e-15, den, jnp.nan)
            return num / den                                # frazione

        def empirical_cvar_jax(L, alpha):
            """
            Versione senza slice dinamici:
            - usa dynamic_index_in_dim per s[k-1]
            - usa mask su indici per sommare la coda (evita s[k:T])
            """
            valid = jnp.isfinite(L)                              # (T,)
            T = jnp.sum(valid).astype(jnp.int32)
            Lc = jnp.where(valid, L, jnp.inf)                   # push non-validi in fondo
            s = jnp.sort(Lc)                                    # (T_total,)
            idx = jnp.arange(s.shape[0], dtype=jnp.int32)

            T_f = T.astype(jnp.float64)
            k = jnp.clip(jnp.ceil(alpha * T_f).astype(jnp.int32), 1, jnp.maximum(T, 1))

            km1 = jnp.maximum(k - 1, 0)
            var_alpha = jnp.where(T > 0, lax.dynamic_index_in_dim(s, km1, keepdims=False), jnp.nan)

            valid_mask = idx < T
            tail_mask = (idx >= k) & valid_mask
            tail_sum = jnp.sum(jnp.where(tail_mask, s, 0.0))

            frac  = k.astype(jnp.float64) - alpha * T_f
            denom = (1.0 - alpha) * T_f
            return jnp.where(T > 0, (frac * var_alpha + tail_sum) / denom, jnp.nan)

        def fin_mean_cvar_emp_utility_jax(x, lmbd, alpha):
            r = portfolio_series_jax(x)
            mean_r = jnp.nanmean(r)
            L = -r
            cvar = empirical_cvar_jax(L, alpha)
            return jnp.where(jnp.isfinite(cvar), (lmbd * mean_r - (1.0 - lmbd) * cvar)*100, -1e300)

        def esg_mean_variance_utility_jax(x, mu, Sigma, lmbd):
            return (lmbd * jnp.dot(x, mu) - (1.0 - lmbd) * jnp.dot(x, Sigma @ x))

        mu_esg_j    = jnp.asarray(mu_esg_np)
        Sigma_esg_j = jnp.asarray(Sigma_esg_np)

        def log_fin(x):
            return fin_mean_cvar_emp_utility_jax(x, lfin, alpha_fin) * nu_fin

        def log_esg(x):
            return esg_mean_variance_utility_jax(x, mu_esg_j, Sigma_esg_j, lesg) * nu_esg

        import inspect as _inspect
        sig = _inspect.signature(expectation_DPU)
        if "rng_seed" in sig.parameters:
            w_DPU = expectation_DPU(k_val, m, log_fin, log_esg, n_assets, rng_seed=rng_seed)
        else:
            w_DPU = expectation_DPU(k_val, m, log_fin, log_esg, n_assets)

        w_DPU = _to_host_np(w_DPU)
        if w_DPU.sum() != 0:
            w_DPU = w_DPU / w_DPU.sum()

        hhi_dpu = float(np.sum(w_DPU**2))

        # ====== return ======
        return {
            "ok": True,
            "m": m, "lmbd_fin": lfin, "lmbd_esg": lesg,
            "k": k_val, "alpha_fin": alpha_fin,
            "hhi_meu": hhi_meu, "hhi_dpu": hhi_dpu,
            "w_MEU": w_MEU, "w_DPU": w_DPU,
            "rng_seed": rng_seed,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": repr(e),
        }
