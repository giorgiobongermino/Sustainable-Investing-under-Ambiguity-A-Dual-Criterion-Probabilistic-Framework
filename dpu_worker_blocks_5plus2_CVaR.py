import os


# === empirical CVaR helper (stable, vectorized) ===
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
    var_alpha = s[k - 1]
    tail_sum = s[k:].sum()
    frac = k - alpha * T
    return float((frac * var_alpha + tail_sum) / ((1.0 - alpha) * T))


def run_one_dpu_job(cfg):
    """
    Variante 5+2 per il caso CVaR empirico lato finanziario.

    Flusso:
      1) Stima pesi su 5 anni (MEU e DPU), con Fin: mean–CVaR empirico; ESG: mean–variance.
      2) Ristima pesi su 7 anni (5 + 2), con le stesse utilità.
      3) Calcola serie e score per 5 anni (in-sample e next2) e per 7 anni (in-sample).

    Chiavi attese in cfg (principali):
      - Parametri: m, lmbd_fin, lmbd_esg, k_5, k_7, n_assets, rng_seed, alpha_fin
      - Statistiche ESG 5 anni: mu_esg_5, Sigma_esg_5, nu_esg_5
      - Statistiche ESG 7 anni: mu_esg_7, Sigma_esg_7, nu_esg_7
      - Finanziario (dataset):
          * fin_df_5 (serie %), fin_np_5 (T x N), fin_mask_5 (T x N)
          * fin_df_7 (serie %), fin_np_7 (T x N), fin_mask_7 (T x N)
          * next2: { FIN_df (serie %), ESG_df }
      - ESG (dataset): esg_df_5, esg_df_7

    Ritorna un dict picklable che include pesi e, se disponibili, serie/score.
    """
    try:
        # ====== ENV prima degli import JAX/NumPyro ======
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        # ====== import ======
        import random as _py_random
        import numpy as np
        import jax, jax.numpy as jnp
        from jax import lax
        from numpyro import set_platform, enable_x64
        set_platform("cpu"); enable_x64()

        from codice_base import (
            expectation_DPU,
            simplex_optimize,
        )

        import numpy as _np
        import pandas as pd

        # ---------- helper: materializza qualunque jax.Array/DeviceArray/Tracer su host NumPy ----------
        def _to_host_np(x, dtype=float):
            try:
                import jax
                if hasattr(x, "block_until_ready"):
                    x = x.block_until_ready()
                x = jax.device_get(x)
            except Exception:
                pass
            return _np.asarray(x, dtype=dtype)

        def esg_mean_variance_utility(x, mu, Sigma, lmbd):
            # stesso schema del worker CVaR esistente: lmbd sulla media, (1-lmbd) sulla varianza
            x = jnp.asarray(x)
            mu = jnp.asarray(mu)
            Sigma = jnp.asarray(Sigma)
            return (lmbd * jnp.dot(x, mu) - (1.0 - lmbd) * jnp.dot(x, Sigma @ x))

        # ---- serie portafoglio empirica e utility Fin mean–CVaR empirico (Python/NumPy) ----
        def _portfolio_series_emp(fin_df: pd.DataFrame, w: _np.ndarray) -> pd.Series:
            W = _np.asarray(w, float).reshape(-1)
            cols = fin_df.columns[: len(W)]
            r = fin_df[cols]
            def _row_dot(row_vals):
                mask = _np.isfinite(row_vals)
                if not mask.any():
                    return _np.nan
                ww = W[:len(row_vals)][mask]
                s = ww.sum()
                if s != 0:
                    ww = ww / s
                return float(_np.dot(row_vals[mask], ww))
            return r.apply(lambda row: _row_dot(row.values.astype(float)), axis=1)

        def fin_mean_cvar_emp_utility(x, fin_df: pd.DataFrame, lmbd: float, alpha: float) -> float:
            port = _portfolio_series_emp(fin_df, _np.asarray(x, float))
            r = _np.asarray(port, float) / 100.0
            L = -r
            # CVaR empirico
            _Ls = _np.sort(_np.asarray(L, float))
            _Ls = _Ls[_np.isfinite(_Ls)]
            _T  = _Ls.size
            if _T == 0:
                cvar_emp = _np.nan
            else:
                _k = int(_np.ceil(alpha * _T))
                _k = max(1, min(_k, _T))
                _var_alpha = _Ls[_k-1]
                _tail_sum  = _Ls[_k:].sum()
                _frac      = _k - alpha * _T
                cvar_emp = float((_frac * _var_alpha + _tail_sum) / ((1.0 - alpha) * _T))
            mean_r = float(_np.nanmean(r))
            return (lmbd * mean_r - (1.0 - lmbd) * cvar_emp)*100

        def double_objective_meanCVaR_fin__meanVar_esg(
            x,
            fin_df: pd.DataFrame,
            mu_esg, Sigma_esg,
            lfin: float, lesg: float,
            m: float, k_val: float,
            alpha_fin: float,
        ) -> float:
            u_fin = fin_mean_cvar_emp_utility(x, fin_df, lfin, alpha_fin)
            u_esg = esg_mean_variance_utility(x, mu_esg, Sigma_esg, lesg)
            return (1.0 - m*k_val) * u_fin + m * k_val *( u_esg)

        # ---- JAX: serie portafoglio e utility Fin mean–CVaR empirico ----
        def _portfolio_returns_jax(fin_vals, fin_mask, w):
            fin = jnp.asarray(fin_vals, dtype=jnp.float64)
            mask_bool = jnp.asarray(fin_mask, dtype=jnp.bool_)
            mask = mask_bool.astype(fin.dtype)
            w = jnp.asarray(w, dtype=fin.dtype)
            w_b = jnp.expand_dims(w, axis=0)
            w_masked = w_b * mask
            denom = jnp.sum(w_masked, axis=1, keepdims=True)
            denom = jnp.where(denom > 0.0, denom, 1.0)
            w_norm = jnp.where(mask_bool, w_masked / denom, 0.0)
            port = jnp.sum(fin * w_norm, axis=1)
            valid_rows = jnp.any(mask_bool, axis=1)
            port = jnp.where(valid_rows, port, jnp.nan)
            return port, valid_rows

        def fin_mean_cvar_emp_utility_jax(x, fin_vals, fin_mask, lmbd: float, alpha: float):
            port, valid_rows = _portfolio_returns_jax(fin_vals, fin_mask, x)
            r = port / 100.0
            valid = jnp.isfinite(r) & valid_rows
            T = jnp.sum(valid)
            dtype = r.dtype
            def _true_branch(_):
                T_f = jnp.asarray(T, dtype=dtype)
                r_sum = jnp.sum(jnp.where(valid, r, 0.0))
                mean_r = r_sum / T_f
                losses = -r
                losses_masked = jnp.where(valid, losses, jnp.inf)
                sorted_losses = jnp.sort(losses_masked)
                idx = jnp.arange(sorted_losses.shape[0], dtype=jnp.int32)
                k_float = jnp.ceil(alpha * T_f)
                k_int = jnp.clip(k_float.astype(jnp.int32), 1, T)
                var_alpha = lax.dynamic_index_in_dim(sorted_losses, k_int - 1, keepdims=False)
                tail_mask = (idx >= k_int) & (idx < T)
                tail_sum = jnp.sum(jnp.where(tail_mask, sorted_losses, 0.0))
                frac = jnp.asarray(k_int, dtype=dtype) - alpha * T_f
                denom = (1.0 - alpha) * T_f
                cvar = (frac * var_alpha + tail_sum) / denom
                return (lmbd * mean_r - (1.0 - lmbd) * cvar)*100

            def _false_branch(_):
                return -jnp.inf

            return lax.cond(T > 0, _true_branch, _false_branch, operand=None)

        def fin_mean_cvar_emp_logp(x, fin_vals, fin_mask, lmbd, alpha, nu=1.0):
            u = fin_mean_cvar_emp_utility_jax(x, fin_vals, fin_mask, lmbd, alpha)
            return u * nu

        def esg_mean_var_logp(x, mu, Sigma, lmbd, nu=1.0):
            u = esg_mean_variance_utility(x, mu, Sigma, lmbd)
            return u * nu

        # ====== cfg ======
        win_tag  = cfg["win_tag"]
        i_m      = int(cfg["i_m"])
        i_f      = int(cfg["i_f"])
        i_e      = int(cfg["i_e"])
        m        = float(cfg["m"])
        lfin     = float(cfg["lmbd_fin"])
        lesg     = float(cfg["lmbd_esg"])
        k_5      = float(cfg["k_5"])
        k_7      = float(cfg["k_7"])
        n_assets = int(cfg["n_assets"])
        rng_seed = int(cfg.get("rng_seed", 0))
        alpha_fin = float(cfg.get("alpha_fin", 0.95))
        alpha_var = float(cfg.get("alpha_var", alpha_fin))

        _py_random.seed(rng_seed)
        np.random.seed(rng_seed % (2**32 - 1))

        # 5 anni (ESG)
        mu_esg_5    = _np.asarray(cfg["mu_esg_5"], dtype=float)
        Sigma_esg_5 = _np.asarray(cfg["Sigma_esg_5"], dtype=float)
        nu_esg_5    = float(cfg.get("nu_esg_5", 1.0))

        # 7 anni (ESG)
        mu_esg_7    = _np.asarray(cfg["mu_esg_7"], dtype=float)
        Sigma_esg_7 = _np.asarray(cfg["Sigma_esg_7"], dtype=float)
        nu_esg_7    = float(cfg.get("nu_esg_7", 1.0))

        # dataset finanziari per JAX logp (necessari)
        fin_np_5   = _np.array(cfg["fin_np_5"], dtype=float, copy=True)
        fin_mask_5 = _np.array(cfg["fin_mask_5"], dtype=bool,  copy=False)
        fin_np_5[~fin_mask_5] = 0.0

        fin_np_7   = _np.array(cfg["fin_np_7"], dtype=float, copy=True)
        fin_mask_7 = _np.array(cfg["fin_mask_7"], dtype=bool,  copy=False)
        fin_np_7[~fin_mask_7] = 0.0

        # dataset per serie e score
        fin_df_5 = cfg.get("fin_df_5", None)
        esg_df_5 = cfg.get("esg_df_5", None)
        nxt = cfg.get("next2", {})
        fin_df_n2 = nxt.get("FIN_df", None)
        esg_df_n2 = nxt.get("ESG_df", None)
        fin_df_7 = cfg.get("fin_df_7", None)
        esg_df_7 = cfg.get("esg_df_7", None)

        # scaling ν
        nu_fin_5 = float(cfg.get("nu_fin_5", 1.0))
        nu_fin_7 = float(cfg.get("nu_fin_7", 1.0))

        # ====== 1) MEU (5 anni) ======
        def MEU5(x):
            return double_objective_meanCVaR_fin__meanVar_esg(
                x, fin_df_5, mu_esg_5, Sigma_esg_5, lfin, lesg, m, k_5, alpha_fin
            )
        w_MEU_5, _, _ = simplex_optimize(MEU5, n_assets)
        w_MEU_5 = _to_host_np(w_MEU_5)
        if w_MEU_5.size and _np.isfinite(w_MEU_5).any():
            s = _np.sum(w_MEU_5)
            if s != 0:
                w_MEU_5 = w_MEU_5 / s

        # ====== 2) DPU (5 anni) ======
        mu_esg_5_j    = jnp.asarray(mu_esg_5)
        Sigma_esg_5_j = jnp.asarray(Sigma_esg_5)
        fin_np_5_j    = jnp.asarray(fin_np_5)
        fin_mask_5_j  = jnp.asarray(fin_mask_5)
        nu_fin_5_j    = jnp.asarray(nu_fin_5, dtype=mu_esg_5_j.dtype)
        nu_esg_5_j    = jnp.asarray(nu_esg_5, dtype=mu_esg_5_j.dtype)

        def log_fin_5(x):
            return fin_mean_cvar_emp_logp(x, fin_np_5_j, fin_mask_5_j, lfin, alpha_fin, nu_fin_5_j)

        def log_esg_5(x):
            return esg_mean_var_logp(x, mu_esg_5_j, Sigma_esg_5_j, lesg, nu_esg_5_j)

        import inspect as _inspect
        sig = _inspect.signature(expectation_DPU)
        if "rng_seed" in sig.parameters:
            w_DPU_5 = expectation_DPU(k_5, m, log_fin_5, log_esg_5, n_assets, rng_seed=rng_seed)
        else:
            w_DPU_5 = expectation_DPU(k_5, m, log_fin_5, log_esg_5, n_assets)
        w_DPU_5 = _to_host_np(w_DPU_5)
        if w_DPU_5.size and _np.isfinite(w_DPU_5).any():
            s = _np.sum(w_DPU_5)
            if s != 0:
                w_DPU_5 = w_DPU_5 / s

        # ====== 3) MEU (7 anni) ======
        def MEU7(x):
            return double_objective_meanCVaR_fin__meanVar_esg(
                x, fin_df_7, mu_esg_7, Sigma_esg_7, lfin, lesg, m, k_7, alpha_fin
            )
        w_MEU_7, _, _ = simplex_optimize(MEU7, n_assets)
        w_MEU_7 = _to_host_np(w_MEU_7)
        if w_MEU_7.size and _np.isfinite(w_MEU_7).any():
            s = _np.sum(w_MEU_7)
            if s != 0:
                w_MEU_7 = w_MEU_7 / s

        # ====== 4) DPU (7 anni) ======
        mu_esg_7_j    = jnp.asarray(mu_esg_7)
        Sigma_esg_7_j = jnp.asarray(Sigma_esg_7)
        fin_np_7_j    = jnp.asarray(fin_np_7)
        fin_mask_7_j  = jnp.asarray(fin_mask_7)
        nu_fin_7_j    = jnp.asarray(nu_fin_7, dtype=mu_esg_7_j.dtype)
        nu_esg_7_j    = jnp.asarray(nu_esg_7, dtype=mu_esg_7_j.dtype)

        def log_fin_7(x):
            return fin_mean_cvar_emp_logp(x, fin_np_7_j, fin_mask_7_j, lfin, alpha_fin, nu_fin_7_j)

        def log_esg_7(x):
            return esg_mean_var_logp(x, mu_esg_7_j, Sigma_esg_7_j, lesg, nu_esg_7_j)

        if "rng_seed" in sig.parameters:
            w_DPU_7 = expectation_DPU(k_7, m, log_fin_7, log_esg_7, n_assets, rng_seed=rng_seed)
        else:
            w_DPU_7 = expectation_DPU(k_7, m, log_fin_7, log_esg_7, n_assets)
        w_DPU_7 = _to_host_np(w_DPU_7)
        if w_DPU_7.size and _np.isfinite(w_DPU_7).any():
            s = _np.sum(w_DPU_7)
            if s != 0:
                w_DPU_7 = w_DPU_7 / s

        # ====== 5) helper per serie e score annuali ======
        def _portfolio_series(fin_df: pd.DataFrame, w: _np.ndarray) -> pd.Series:
            if fin_df is None or fin_df.empty:
                return pd.Series(dtype=float)
            W = _np.asarray(w, float).reshape(-1)
            cols = fin_df.columns[: len(W)]
            r = fin_df[cols]
            def _row_dot(row_vals):
                mask = _np.isfinite(row_vals)
                if not mask.any():
                    return _np.nan
                ww = W[:len(row_vals)][mask]
                ww = ww / ww.sum() if ww.sum() != 0 else ww
                return float(_np.dot(row_vals[mask], ww))
            return r.apply(lambda row: _row_dot(row.values.astype(float)), axis=1)

        def _annual_scores(port_series: pd.Series, esg_df: pd.DataFrame, w: _np.ndarray,
                           m: float, lfin: float, k: float, alpha_fin: float) -> dict:
            out = {}
            if port_series is None or port_series.empty or esg_df is None or esg_df.empty:
                return out
            esg_cols = esg_df.columns[: len(w)]
            esg_port_year = (esg_df[esg_cols] @ _np.asarray(w[:len(esg_cols)], float)).to_dict()
            df = port_series.to_frame("r")
            df["year"] = df.index.year
            grp = df.groupby("year")["r"]
            mean_r = grp.mean()
            # stima cvar annuo empirico sul set di rendimenti dell'anno
            cvar_by_year = {}
            for y, series in grp:
                r_y = _np.asarray(series, float) / 100.0
                cvar_by_year[int(y)] = _empirical_cvar_from_losses(-r_y, alpha_fin)
            for y in mean_r.index:
                er = float(mean_r.loc[y]) / 100.0
                cvar_y = float(cvar_by_year.get(int(y), _np.nan))
                esg_y = float(esg_port_year.get(int(y), _np.nan))
                out[int(y)] = (1.0 - m*k) * (er - lfin * cvar_y) + m * (k * esg_y)
            return out

        def _cum_from_simple(port_series):
            if port_series is None or port_series.empty:
                return pd.Series(dtype=float)
            return (1.0 + port_series.div(100.0)).cumprod().sub(1.0).mul(100.0)

        def _var_cvar(port_series, alpha: float):
            """
            VaR e CVaR empirici sulle perdite (-r) al livello 'alpha'.
            Restituisce None se la serie non è disponibile.
            """
            if port_series is None or port_series.empty:
                return None
            vals = _np.asarray(port_series, float)
            vals = vals[_np.isfinite(vals)]
            if vals.size == 0:
                return None
            losses = -vals
            losses = _np.sort(losses)
            T = losses.size
            k = int(_np.ceil(alpha * T))
            k = max(1, min(k, T))
            var_alpha = losses[k - 1]
            tail_sum = losses[k:].sum()
            frac = k - alpha * T
            cvar = (frac * var_alpha + tail_sum) / ((1.0 - alpha) * T)
            return {"VaR": float(var_alpha), "CVaR": float(cvar)}

        def _esg_port_series_yearly(esg_df: pd.DataFrame, w: _np.ndarray) -> pd.Series:
            if esg_df is None or esg_df.empty:
                return pd.Series(dtype=float, name="esg_port")
            esg_cols = esg_df.columns[: len(w)]
            w_clip = _np.asarray(w[:len(esg_cols)], float)
            vals = (esg_df[esg_cols].to_numpy(dtype=float) @ w_clip)
            idx = esg_df.index
            try:
                years = pd.Index(pd.to_datetime(idx)).year.astype(int)
            except Exception:
                years = pd.Index(_np.asarray(idx, int))
            return pd.Series(vals, index=years, name="esg_port")

        # ====== 6) serie/score 5 anni e 7 anni ======
        # 5 anni (pesi 5)
        perf_MEU5_in = perf_DPU5_in = perf_MEU5_in_cum = perf_DPU5_in_cum = None
        perf_MEU5_next2 = perf_DPU5_next2 = perf_MEU5_next2_cum = perf_DPU5_next2_cum = None
        score_MEU5_in = score_DPU5_in = {}
        score_MEU5_next2 = score_DPU5_next2 = {}
        esg_MEU5_yearly_in = esg_DPU5_yearly_in = None
        esg_MEU5_yearly_next2 = esg_DPU5_yearly_next2 = None
        risk_MEU5_in = risk_DPU5_in = None
        risk_MEU5_next2 = risk_DPU5_next2 = None

        if fin_df_5 is not None and esg_df_5 is not None:
            perf_MEU5_in = _portfolio_series(fin_df_5, w_MEU_5)
            perf_DPU5_in = _portfolio_series(fin_df_5, w_DPU_5)
            perf_MEU5_in_cum = _cum_from_simple(perf_MEU5_in)
            perf_DPU5_in_cum = _cum_from_simple(perf_DPU5_in)
            score_MEU5_in = _annual_scores(perf_MEU5_in, esg_df_5, w_MEU_5, m, lfin, k_5, alpha_fin)
            score_DPU5_in = _annual_scores(perf_DPU5_in, esg_df_5, w_DPU_5, m, lfin, k_5, alpha_fin)
            esg_MEU5_yearly_in = _esg_port_series_yearly(esg_df_5, w_MEU_5)
            esg_DPU5_yearly_in = _esg_port_series_yearly(esg_df_5, w_DPU_5)
            risk_MEU5_in = _var_cvar(perf_MEU5_in, alpha_var)
            risk_DPU5_in = _var_cvar(perf_DPU5_in, alpha_var)

        if fin_df_n2 is not None and esg_df_n2 is not None:
            perf_MEU5_next2 = _portfolio_series(fin_df_n2, w_MEU_5)
            perf_DPU5_next2 = _portfolio_series(fin_df_n2, w_DPU_5)
            perf_MEU5_next2_cum = _cum_from_simple(perf_MEU5_next2)
            perf_DPU5_next2_cum = _cum_from_simple(perf_DPU5_next2)
            score_MEU5_next2 = _annual_scores(perf_MEU5_next2, esg_df_n2, w_MEU_5, m, lfin, k_5, alpha_fin)
            score_DPU5_next2 = _annual_scores(perf_DPU5_next2, esg_df_n2, w_DPU_5, m, lfin, k_5, alpha_fin)
            esg_MEU5_yearly_next2 = _esg_port_series_yearly(esg_df_n2, w_MEU_5)
            esg_DPU5_yearly_next2 = _esg_port_series_yearly(esg_df_n2, w_DPU_5)
            risk_MEU5_next2 = _var_cvar(perf_MEU5_next2, alpha_var)
            risk_DPU5_next2 = _var_cvar(perf_DPU5_next2, alpha_var)

        # 7 anni (pesi 7)
        perf_MEU7_in = perf_DPU7_in = perf_MEU7_in_cum = perf_DPU7_in_cum = None
        score_MEU7_in = score_DPU7_in = {}
        esg_MEU7_yearly_in = esg_DPU7_yearly_in = None
        risk_MEU7_in = risk_DPU7_in = None

        if fin_df_7 is not None and esg_df_7 is not None:
            perf_MEU7_in = _portfolio_series(fin_df_7, w_MEU_7)
            perf_DPU7_in = _portfolio_series(fin_df_7, w_DPU_7)
            perf_MEU7_in_cum = _cum_from_simple(perf_MEU7_in)
            perf_DPU7_in_cum = _cum_from_simple(perf_DPU7_in)
            score_MEU7_in = _annual_scores(perf_MEU7_in, esg_df_7, w_MEU_7, m, lfin, k_7, alpha_fin)
            score_DPU7_in = _annual_scores(perf_DPU7_in, esg_df_7, w_DPU_7, m, lfin, k_7, alpha_fin)
            esg_MEU7_yearly_in = _esg_port_series_yearly(esg_df_7, w_MEU_7)
            esg_DPU7_yearly_in = _esg_port_series_yearly(esg_df_7, w_DPU_7)
            risk_MEU7_in = _var_cvar(perf_MEU7_in, alpha_var)
            risk_DPU7_in = _var_cvar(perf_DPU7_in, alpha_var)

        # ====== 7) return ======
        res = {
            "ok": True,
            "win_tag": win_tag,
            "i_m": i_m, "i_f": i_f, "i_e": i_e,
            "m": m, "lmbd_fin": lfin, "lmbd_esg": lesg,
            "alpha_fin": alpha_fin,
            "alpha_var": alpha_var,
            "k_5": k_5, "k_7": k_7,
            "w_MEU_5": w_MEU_5, "w_DPU_5": w_DPU_5,
            "w_MEU_7": w_MEU_7, "w_DPU_7": w_DPU_7,
            "rng_seed": rng_seed,
        }

        # 5 anni
        res.update({
            "perf_MEU5_in": perf_MEU5_in,
            "perf_DPU5_in": perf_DPU5_in,
            "perf_MEU5_in_cum": perf_MEU5_in_cum,
            "perf_DPU5_in_cum": perf_DPU5_in_cum,
            "perf_MEU5_next2": perf_MEU5_next2,
            "perf_DPU5_next2": perf_DPU5_next2,
            "perf_MEU5_next2_cum": perf_MEU5_next2_cum,
            "perf_DPU5_next2_cum": perf_DPU5_next2_cum,
            "score_MEU5_in": score_MEU5_in,
            "score_DPU5_in": score_DPU5_in,
            "score_MEU5_next2": score_MEU5_next2,
            "score_DPU5_next2": score_DPU5_next2,
            "esg_MEU5_yearly_in": esg_MEU5_yearly_in,
            "esg_DPU5_yearly_in": esg_DPU5_yearly_in,
            "esg_MEU5_yearly_next2": esg_MEU5_yearly_next2,
            "esg_DPU5_yearly_next2": esg_DPU5_yearly_next2,
            "risk_MEU5_in": risk_MEU5_in,
            "risk_DPU5_in": risk_DPU5_in,
            "risk_MEU5_next2": risk_MEU5_next2,
            "risk_DPU5_next2": risk_DPU5_next2,
        })

        # 7 anni
        res.update({
            "perf_MEU7_in": perf_MEU7_in,
            "perf_DPU7_in": perf_DPU7_in,
            "perf_MEU7_in_cum": perf_MEU7_in_cum,
            "perf_DPU7_in_cum": perf_DPU7_in_cum,
            "score_MEU7_in": score_MEU7_in,
            "score_DPU7_in": score_DPU7_in,
            "esg_MEU7_yearly_in": esg_MEU7_yearly_in,
            "esg_DPU7_yearly_in": esg_DPU7_yearly_in,
            "risk_MEU7_in": risk_MEU7_in,
            "risk_DPU7_in": risk_DPU7_in,
        })

        return res

    except Exception as e:
        return {
            "ok": False,
            "win_tag": cfg.get("win_tag"),
            "i_m": cfg.get("i_m"), "i_f": cfg.get("i_f"), "i_e": cfg.get("i_e"),
            "error": repr(e),
        }
