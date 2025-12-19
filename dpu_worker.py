# dpu_worker.py
import os

def run_one_dpu_job(cfg):
    """
    Esegue in UN processo:
      1) MEU via simplex_optimize / double_mean_variance_objective (NumPy/SciPy)
      2) DPU via expectation_DPU (NumPyro/JAX)
    Ritorna metriche + pesi e la chiave 'key'.
    """
    try:
        # ====== ENV: prima degli import JAX/NumPyro ======
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # single-thread XLA/Eigen per evitare over-subscription quando lanci molti processi
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
        # limita BLAS/NumExpr a 1 thread
        os.environ["OMP_NUM_THREADS"]      = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"]      = "1"
        os.environ["NUMEXPR_NUM_THREADS"]  = "1"

        # ====== import ======
        import numpy as np
        import jax, jax.numpy as jnp
        from numpyro import set_platform, enable_x64
        set_platform("cpu"); enable_x64()

        from codice_base import (
            expectation_DPU,
            simplex_optimize,
            double_mean_variance_objective,
        )

        # ----- helpers -----
        def hhi_raw(w: np.ndarray) -> float:
            w = np.asarray(w, dtype=float); return float(np.sum(w**2))

        def hhi_norm(w: np.ndarray) -> float:
            w = np.asarray(w, dtype=float); s = w.sum()
            return float(np.sum(w**2) / (s*s + 1e-16))

        def mean_var_logp(x, mu, Sigma, lmbd, nu=1.0):
            """Log-utility media-varianza con scala esterna ν (tutto in JAX)."""
            u = lmbd*jnp.dot(x, mu) - (1-lmbd) * jnp.dot(x, Sigma @ x)
            return u * nu

        # ====== estrai cfg ======
        key      = cfg.get("key", "")
        m        = float(cfg["m"])
        lfin     = float(cfg["lmbd_fin"])
        lesg     = float(cfg["lmbd_esg"])
        k_val    = float(cfg["k"])
        n_assets = int(cfg["n_assets"])

        import numpy as _np
        mu_fin_np    = _np.asarray(cfg["mu_fin_np"], dtype=float)
        mu_esg_np    = _np.asarray(cfg["mu_esg_np"], dtype=float)
        Sigma_fin_np = _np.asarray(cfg["Sigma_fin_np"], dtype=float)
        Sigma_esg_np = _np.asarray(cfg["Sigma_esg_np"], dtype=float)

        nu_fin = float(cfg.get("nu_fin", 1.0))
        nu_esg = float(cfg.get("nu_esg", 1.0))

        # opzionale: seed (usa solo se expectation_DPU lo supporta)
        rng_seed = int(cfg.get("rng_seed", 0))

        # ====== 1) MEU (NumPy/SciPy) ======
        def MEU_overall(x):
            return double_mean_variance_objective(
                x,
                mu_fin_np, mu_esg_np,
                Sigma_fin_np, Sigma_esg_np,
                lfin, lesg, m, k_val
            )

        w_MEU, fval_MEU, res_MEU = simplex_optimize(MEU_overall, n_assets)
        if w_MEU.sum() != 0:
            w_MEU = w_MEU / w_MEU.sum()

        hhi_meu      = hhi_raw(w_MEU)
        hhi_meu_norm = hhi_norm(w_MEU)

        # ====== 2) DPU (NumPyro/JAX) ======
        mu_fin_j    = jnp.asarray(mu_fin_np)
        mu_esg_j    = jnp.asarray(mu_esg_np)
        Sigma_fin_j = jnp.asarray(Sigma_fin_np)
        Sigma_esg_j = jnp.asarray(Sigma_esg_np)
        nu_fin_j    = jnp.asarray(nu_fin, dtype=mu_fin_j.dtype)
        nu_esg_j    = jnp.asarray(nu_esg, dtype=mu_esg_j.dtype)

        def log_fin(x): return mean_var_logp(x, mu_fin_j, Sigma_fin_j, lfin, nu_fin_j)
        def log_esg(x): return mean_var_logp(x, mu_esg_j, Sigma_esg_j, lesg, nu_esg_j)

        # se la tua expectation_DPU supporta rng_seed=..., passalo; altrimenti lascia la versione base
        w_DPU = expectation_DPU(k_val, m, log_fin, log_esg, n_assets)  # , rng_seed=rng_seed
        w_DPU = _np.asarray(w_DPU, dtype=float)
        if w_DPU.sum() != 0:
            w_DPU = w_DPU / w_DPU.sum()

        hhi_dpu      = hhi_raw(w_DPU)
        hhi_dpu_norm = hhi_norm(w_DPU)

        return {
            "ok": True,
            "key": key,
            "m": m, "lmbd_fin": lfin, "lmbd_esg": lesg,
            "hhi_meu": hhi_meu, "hhi_meu_norm": hhi_meu_norm,
            "hhi_dpu": hhi_dpu, "hhi_dpu_norm": hhi_dpu_norm,
            "sum_w_meu": float(w_MEU.sum()), "sum_w_dpu": float(w_DPU.sum()),
            "success_meu": bool(getattr(res_MEU, "success", True)),
            "w_MEU": w_MEU, "w_DPU": w_DPU,
        }

    except Exception as e:
        return {
            "ok": False, "error": repr(e),
            "key": cfg.get("key", ""),
            "m": cfg.get("m"), "lmbd_fin": cfg.get("lmbd_fin"), "lmbd_esg": cfg.get("lmbd_esg"),
        }
