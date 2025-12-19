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
    Esegue MEU e DPU e (se presenti in cfg) restituisce:
      - serie storiche settimanali in-sample e next2
      - score annuali in-sample e next2
      - serie storica ESG annuale del portafoglio (MEU e DPU) in-sample e next2

    Differenze vs originale:
      - Parte finanziaria: mean–CVaR EMPIRICO (quantile delle perdite settimanali del portafoglio
        calcolato su fin_df in-sample).
      - Parte ESG: rimane mean–variance (media-varianza) con (mu_esg, Sigma_esg, λ_esg).
    """
    try:
        # ====== ENV prima degli import JAX/NumPyro ======
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # Abilita cache persistente XLA per riusare compilazioni tra run/processi
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

        from codice_base import (
            expectation_DPU,
            simplex_optimize,
        )

        # =======================
        # Helper: utility ESG mean-variance e Fin mean–CVaR empirico
        # =======================
        import numpy as _np
        import pandas as pd

        # ---------- helper: materializza qualunque jax.Array/DeviceArray/Tracer su host NumPy ----------
        def _to_host_np(x, dtype=float):
            """
            Converte jax.Array/DeviceArray/Tracer in NumPy host.
            Non fa nulla se x è già array/lista/float Python.
            """
            try:
                import jax
                if hasattr(x, "block_until_ready"):
                    x = x.block_until_ready()
                x = jax.device_get(x)
            except Exception:
                # jax non disponibile o x già host
                pass
            return _np.asarray(x, dtype=dtype)

        def esg_mean_variance_utility(x, mu, Sigma, lmbd):
            """
            U_ESG(x) = x' mu - 0.5 * lmbd * x' Sigma x
            """
            x = jnp.asarray(x)
            mu = jnp.asarray(mu)
            Sigma = jnp.asarray(Sigma)
            return (lmbd*jnp.dot(x, mu) - (1- lmbd) * jnp.dot(x, Sigma @ x))

        def _portfolio_series_emp(fin_df: pd.DataFrame, w: _np.ndarray) -> pd.Series:
            """
            Serie settimanale del portafoglio, rinormalizzando i pesi sulle sole colonne non-NaN per riga.
            Ritorna rendimenti nello stesso formato di fin_df (tipicamente %).
            """
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
            """
            Utility finanziaria mean–CVaR *empirico*:
              U_fin(x) = mean_r - lmbd * VaR_alpha_empirico(L), con L = -r_port.
            mean/CVaR stimati sui rendimenti settimanali del portafoglio ottenuti da fin_df (in % -> in unità).
            """
            # Serie portafoglio (%)
            port = _portfolio_series_emp(fin_df, _np.asarray(x, float))
            r = _np.asarray(port, float) / 100.0  # in unità
            L = -r
            # CVaR empirico (sostituisce il CVaR)
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
            return (lmbd*mean_r -(1- lmbd) * cvar_emp)*100

        def double_objective_meanCVaR_fin__meanVar_esg(
            x,
            fin_df: pd.DataFrame,
            mu_esg, Sigma_esg,
            lfin: float, lesg: float,
            m: float, k_val: float,
            alpha_fin: float,
        ) -> float:
            """
            Obiettivo complessivo:
              (1-m) * U_fin_empirico(x) + m * [ k * U_ESG_meanVar(x) ]
            """
            u_fin = fin_mean_cvar_emp_utility(x, fin_df, lfin, alpha_fin)
            u_esg = esg_mean_variance_utility(x, mu_esg, Sigma_esg, lesg)
            return ((1.0 - m*k_val) * u_fin + m * k_val *( u_esg))

        # ----- helper (per DPU: “logp” finanziario/ESG coerenti con l’obiettivo sopra) -----
        def _portfolio_returns_jax(fin_vals, fin_mask, w):
            """
            Serie settimanale del portafoglio in JAX, con rinormalizzazione dei pesi
            sulle sole colonne disponibili in ciascuna riga.
            """
            fin = jnp.asarray(fin_vals, dtype=jnp.float64)
            mask_bool = jnp.asarray(fin_mask, dtype=jnp.bool_)
            mask = mask_bool.astype(fin.dtype)
            w = jnp.asarray(w, dtype=fin.dtype)
            w_b = jnp.expand_dims(w, axis=0)  # (1, n_assets)
            w_masked = w_b * mask             # (T, n_assets)

            denom = jnp.sum(w_masked, axis=1, keepdims=True)
            denom = jnp.where(denom > 0.0, denom, 1.0)

            w_norm = jnp.where(mask_bool, w_masked / denom, 0.0)
            port = jnp.sum(fin * w_norm, axis=1)
            valid_rows = jnp.any(mask_bool, axis=1)
            port = jnp.where(valid_rows, port, jnp.nan)
            return port, valid_rows

        def fin_mean_cvar_emp_utility_jax(x, fin_vals, fin_mask, lmbd: float, alpha: float):
            """
            Controparte JAX della mean–CVaR empirica. Restituisce scalar jnp.float64.
            """
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

                return (lmbd*mean_r - (1- lmbd) * cvar)*100

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
        ws       = int(cfg["ws"])
        win_tag  = cfg["win_tag"]
        i_m      = int(cfg["i_m"])
        i_f      = int(cfg["i_f"])
        i_e      = int(cfg["i_e"])
        m        = float(cfg["m"])
        lfin     = float(cfg["lmbd_fin"])
        lesg     = float(cfg["lmbd_esg"])
        k_val    = float(cfg["k"])
        n_assets = int(cfg["n_assets"])
        rng_seed = int(cfg.get("rng_seed", 0))
        alpha_fin = float(cfg.get("alpha_fin", 0.95))  # livello CVaR empirico
        alpha_var = float(cfg.get("alpha_var", alpha_fin))

        _py_random.seed(rng_seed)
        np.random.seed(rng_seed % (2**32 - 1))

        mu_fin_np    = _np.asarray(cfg.get("mu_fin_np", []), dtype=float)
        mu_esg_np    = _np.asarray(cfg["mu_esg_np"], dtype=float)
        Sigma_fin_np = _np.asarray(cfg.get("Sigma_fin_np", []), dtype=float)
        Sigma_esg_np = _np.asarray(cfg["Sigma_esg_np"], dtype=float)

        nu_fin = float(cfg.get("nu_fin", 1.0))
        nu_esg = float(cfg.get("nu_esg", 1.0))

        # ====== dati in-sample (necessari per CVaR empirico fin) ======
        fin_df = cfg.get("fin_df", None)   # DataFrame rendimenti settimanali in-sample
        esg_df = cfg.get("esg_df", None)   # DataFrame ESG annuale della finestra
        if fin_df is None:
            raise ValueError("Per il mean–CVaR empirico finanziario è necessario 'fin_df' in-sample nella cfg.")

        # ====== dataset numerici per componente JAX ======
        fin_np_vals = cfg.get("fin_np", None)
        fin_mask_vals = cfg.get("fin_mask", None)
        if fin_np_vals is None or fin_mask_vals is None:
            raise ValueError("La cfg deve contenere 'fin_np' e 'fin_mask' per il logp finanziario.")
        fin_np_vals = _np.array(fin_np_vals, dtype=float, copy=True)
        fin_mask_vals = _np.array(fin_mask_vals, dtype=bool, copy=False)
        fin_np_vals[~fin_mask_vals] = 0.0

        # ====== 1) MEU ======
        def MEU_overall(x):
            return double_objective_meanCVaR_fin__meanVar_esg(
                x, fin_df, mu_esg_np, Sigma_esg_np, lfin, lesg, m, k_val, alpha_fin
            )

        w_MEU, _, _ = simplex_optimize(MEU_overall, n_assets)
        # --- materializza PRIMA di manipolare con NumPy ---
        w_MEU = _to_host_np(w_MEU)
        if w_MEU.size and _np.isfinite(w_MEU).any():
            s = _np.sum(w_MEU)
            if s != 0:
                w_MEU = w_MEU / s

        # ====== 2) DPU ======
        mu_esg_j    = jnp.asarray(mu_esg_np)
        Sigma_esg_j = jnp.asarray(Sigma_esg_np)
        fin_np_j    = jnp.asarray(fin_np_vals)
        fin_mask_j  = jnp.asarray(fin_mask_vals)
        nu_fin_j    = jnp.asarray(nu_fin, dtype=mu_esg_j.dtype)
        nu_esg_j    = jnp.asarray(nu_esg, dtype=mu_esg_j.dtype)

        # logp finanziario: mean–CVaR empirico (JAX); logp ESG: mean–variance
        def log_fin(x):
            return fin_mean_cvar_emp_logp(x, fin_np_j, fin_mask_j, lfin, alpha_fin, nu_fin_j)
        def log_esg(x): return esg_mean_var_logp(x, mu_esg_j, Sigma_esg_j, lesg, nu_esg_j)

        import inspect as _inspect
        sig = _inspect.signature(expectation_DPU)
        if "rng_seed" in sig.parameters:
            w_DPU = expectation_DPU(k_val, m, log_fin, log_esg, n_assets, rng_seed=rng_seed)
        else:
            w_DPU = expectation_DPU(k_val, m, log_fin, log_esg, n_assets)

        # --- materializza PRIMA di manipolare con NumPy ---
        w_DPU = _to_host_np(w_DPU)
        if w_DPU.size and _np.isfinite(w_DPU).any():
            s = _np.sum(w_DPU)
            if s != 0:
                w_DPU = w_DPU / s

        # ====== 3) helper per serie settimanali e score annuali ======
        def _portfolio_series(fin_df: pd.DataFrame, w: _np.ndarray) -> pd.Series:
            """Ritorna serie settimanale del portafoglio con rinormalizzazione su colonne non-NaN a ogni riga."""
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
                           m: float, lfin: float, k: float, alpha: float = 0.95) -> dict:
            """
            Score annuale:
              (1-m) * ( mean_r - lfin * VaR_alpha_empirico(L) ) + m * ( k * w' ESG_year )
            """
            esg_cols = esg_df.columns[: len(w)]
            esg_port_year = (esg_df[esg_cols] @ _np.asarray(w[:len(esg_cols)], float)).to_dict()
            out = {}
            if port_series is None or port_series.empty:
                return out
            df = port_series.to_frame("r").copy()
            df["year"] = df.index.year
            for y, sub in df.groupby("year"):
                r = _np.asarray(sub["r"].astype(float) / 100.0)
                if r.size == 0:
                    continue
                L = -r
                # CVaR empirico (sostituisce il CVaR)
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
                e_y = esg_port_year.get(int(y), _np.nan)
                if _np.isnan(e_y):
                    e_y = esg_port_year.get(pd.Timestamp(year=int(y), month=1, day=1), _np.nan)
                esg_term = (k * (e_y if _np.isfinite(e_y) else 0.0))
                s_y = (1.0 - m) * (mean_r - lfin * cvar_emp) + m * esg_term
                out[int(y)] = float(s_y)
            return out

        def _cum_from_simple(series: pd.Series) -> pd.Series:
            """Trasforma rendimenti semplici settimanali (%) in cumulata (%) con compounding."""
            s = series.astype(float) / 100.0
            return (1.0 + s).cumprod().sub(1.0).mul(100.0)

        def _var_cvar(port_series: pd.Series, alpha: float):
            """
            VaR e CVaR empirici sulle perdite (-r) al livello 'alpha'.
            Ritorna None se non ci sono valori finiti.
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

        # ====== 3.bis) Serie ESG annuale del portafoglio ======
        def _esg_port_series_yearly(esg_df: pd.DataFrame, w: _np.ndarray) -> pd.Series:
            """
            Serie annuale dell'ESG del portafoglio: w' * ESG_year.
            Restituisce una pd.Series indicizzata per anno (int o derivato da DatetimeIndex).
            """
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

        # ====== 4) calcoli in-sample (se disponibili) ======
        perf_MEU_in = perf_DPU_in = perf_MEU_in_cum = perf_DPU_in_cum = None
        score_MEU_yearly_in = score_DPU_yearly_in = {}
        esg_MEU_yearly_in = esg_DPU_yearly_in = None
        risk_MEU_in = risk_DPU_in = None

        if fin_df is not None and esg_df is not None:
            perf_MEU_in = _portfolio_series(fin_df, w_MEU)
            perf_DPU_in = _portfolio_series(fin_df, w_DPU)
            perf_MEU_in_cum = _cum_from_simple(perf_MEU_in)
            perf_DPU_in_cum = _cum_from_simple(perf_DPU_in)
            score_MEU_yearly_in = _annual_scores(perf_MEU_in, esg_df, w_MEU, m, lfin, k_val, alpha_fin)
            score_DPU_yearly_in = _annual_scores(perf_DPU_in, esg_df, w_DPU, m, lfin, k_val, alpha_fin)

            # ESG annuale (MEU/DPU) in-sample
            esg_MEU_yearly_in = _esg_port_series_yearly(esg_df, w_MEU)
            esg_DPU_yearly_in = _esg_port_series_yearly(esg_df, w_DPU)
            risk_MEU_in = _var_cvar(perf_MEU_in, alpha_var)
            risk_DPU_in = _var_cvar(perf_DPU_in, alpha_var)

        # ====== 5) calcoli sui 2 anni successivi (se presenti) ======
        perf_MEU_next2 = perf_DPU_next2 = perf_MEU_next2_cum = perf_DPU_next2_cum = None
        score_MEU_yearly_next2 = score_DPU_yearly_next2 = {}
        esg_MEU_yearly_next2 = esg_DPU_yearly_next2 = None
        risk_MEU_next2 = risk_DPU_next2 = None

        nxt = cfg.get("next2", {})
        fin_df_n2 = nxt.get("FIN_df", None)
        esg_df_n2 = nxt.get("ESG_df", None)
        if fin_df_n2 is not None and esg_df_n2 is not None:
            perf_MEU_next2 = _portfolio_series(fin_df_n2, w_MEU)
            perf_DPU_next2 = _portfolio_series(fin_df_n2, w_DPU)
            perf_MEU_next2_cum = _cum_from_simple(perf_MEU_next2)
            perf_DPU_next2_cum = _cum_from_simple(perf_DPU_next2)
            score_MEU_yearly_next2 = _annual_scores(perf_MEU_next2, esg_df_n2, w_MEU, m, lfin, k_val, alpha_fin)
            score_DPU_yearly_next2 = _annual_scores(perf_DPU_next2, esg_df_n2, w_DPU, m, lfin, k_val, alpha_fin)

            esg_MEU_yearly_next2 = _esg_port_series_yearly(esg_df_n2, w_MEU)
            esg_DPU_yearly_next2 = _esg_port_series_yearly(esg_df_n2, w_DPU)
            risk_MEU_next2 = _var_cvar(perf_MEU_next2, alpha_var)
            risk_DPU_next2 = _var_cvar(perf_DPU_next2, alpha_var)

        # ====== 6) return ======
        res = {
            "ok": True,
            "ws": ws, "win_tag": win_tag,
            "i_m": i_m, "i_f": i_f, "i_e": i_e,
            "m": m, "lmbd_fin": lfin, "lmbd_esg": lesg,
            "k": k_val,
            "alpha_fin": alpha_fin,
            "alpha_var": alpha_var,
            "w_MEU": w_MEU, "w_DPU": w_DPU,
            "rng_seed": rng_seed,
        }

        # aggiungi serie e score solo se calcolati
        if fin_df is not None and esg_df is not None:
            res.update({
                "perf_MEU_in": perf_MEU_in,                # pd.Series (weekly %)
                "perf_DPU_in": perf_DPU_in,
                "perf_MEU_in_cum": perf_MEU_in_cum,        # pd.Series (cumulative %)
                "perf_DPU_in_cum": perf_DPU_in_cum,
                "score_MEU_yearly_in": score_MEU_yearly_in,  # dict {year: score}
                "score_DPU_yearly_in": score_DPU_yearly_in,

                # ESG annuale del portafoglio (MEU/DPU) in-sample
                "esg_MEU_yearly_in": esg_MEU_yearly_in,      # pd.Series (annuale)
                "esg_DPU_yearly_in": esg_DPU_yearly_in,      # pd.Series (annuale)
                "risk_MEU_in": risk_MEU_in,
                "risk_DPU_in": risk_DPU_in,
            })
        if fin_df_n2 is not None and esg_df_n2 is not None:
            res.update({
                "perf_MEU_next2": perf_MEU_next2,
                "perf_DPU_next2": perf_DPU_next2,
                "perf_MEU_next2_cum": perf_MEU_next2_cum,
                "perf_DPU_next2_cum": perf_DPU_next2_cum,
                "score_MEU_yearly_next2": score_MEU_yearly_next2,
                "score_DPU_yearly_next2": score_DPU_yearly_next2,

                # ESG annuale del portafoglio (MEU/DPU) next2
                "esg_MEU_yearly_next2": esg_MEU_yearly_next2,  # pd.Series (annuale)
                "esg_DPU_yearly_next2": esg_DPU_yearly_next2,  # pd.Series (annuale)
                "risk_MEU_next2": risk_MEU_next2,
                "risk_DPU_next2": risk_DPU_next2,
            })

        return res

    except Exception as e:
        return {
            "ok": False,
            "ws": cfg.get("ws"), "win_tag": cfg.get("win_tag"),
            "i_m": cfg.get("i_m"), "i_f": cfg.get("i_f"), "i_e": cfg.get("i_e"),
            "error": repr(e),
        }
