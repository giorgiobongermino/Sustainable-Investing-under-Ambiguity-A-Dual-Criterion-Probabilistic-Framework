import os

def run_one_dpu_job(cfg):
    """
    Esegue MEU e DPU e (se presenti in cfg) restituisce:
      - serie storiche settimanali in-sample e next2
      - score annuali in-sample e next2
      - NEW: serie storica ESG annuale del portafoglio (MEU e DPU) in-sample e next2
    """
    try:
        # ====== ENV prima degli import JAX/NumPyro ======
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # Abilita cache persistente XLA e preserva flag esistenti
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
        os.environ["OMP_NUM_THREADS"]      = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"]      = "1"
        os.environ["NUMEXPR_NUM_THREADS"]  = "1"

        # ====== import ======
        import random as _py_random
        import numpy as np
        import jax, jax.numpy as jnp
        from numpyro import set_platform, enable_x64
        set_platform("cpu"); enable_x64()

        from codice_base import (
            expectation_DPU,
            simplex_optimize,
            double_mean_variance_objective,
        )

        # ----- helper (scala esterna ν) -----
        def mean_var_logp(x, mu, Sigma, lmbd, nu=1.0):
            u = lmbd*jnp.dot(x, mu) - (1-lmbd) * jnp.dot(x, Sigma @ x)
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

        _py_random.seed(rng_seed)
        np.random.seed(rng_seed % (2**32 - 1))

        import numpy as _np
        mu_fin_np    = _np.asarray(cfg["mu_fin_np"], dtype=float)
        mu_esg_np    = _np.asarray(cfg["mu_esg_np"], dtype=float)
        Sigma_fin_np = _np.asarray(cfg["Sigma_fin_np"], dtype=float)
        Sigma_esg_np = _np.asarray(cfg["Sigma_esg_np"], dtype=float)

        nu_fin = float(cfg.get("nu_fin", 1.0))
        nu_esg = float(cfg.get("nu_esg", 1.0))
        alpha_var = float(cfg.get("alpha_var", 0.95))

        # ====== 1) MEU ======
        def MEU_overall(x):
            return double_mean_variance_objective(
                x, mu_fin_np, mu_esg_np, Sigma_fin_np, Sigma_esg_np,
                lfin, lesg, m, k_val
            )
        w_MEU, _, _ = simplex_optimize(MEU_overall, n_assets)
        if w_MEU.sum() != 0:
            w_MEU = w_MEU / w_MEU.sum()

        # ====== 2) DPU ======
        mu_fin_j    = jnp.asarray(mu_fin_np)
        mu_esg_j    = jnp.asarray(mu_esg_np)
        Sigma_fin_j = jnp.asarray(Sigma_fin_np)
        Sigma_esg_j = jnp.asarray(Sigma_esg_np)
        nu_fin_j    = jnp.asarray(nu_fin, dtype=mu_fin_j.dtype)
        nu_esg_j    = jnp.asarray(nu_esg, dtype=mu_esg_j.dtype)

        def log_fin(x): return mean_var_logp(x, mu_fin_j, Sigma_fin_j, lfin, nu_fin_j)
        def log_esg(x): return mean_var_logp(x, mu_esg_j, Sigma_esg_j, lesg, nu_esg_j)

        import inspect as _inspect
        sig = _inspect.signature(expectation_DPU)
        if "rng_seed" in sig.parameters:
            w_DPU = expectation_DPU(k_val, m, log_fin, log_esg, n_assets, rng_seed=rng_seed)
        else:
            w_DPU = expectation_DPU(k_val, m, log_fin, log_esg, n_assets)

        w_DPU = _np.asarray(w_DPU, dtype=float)
        if w_DPU.sum() != 0:
            w_DPU = w_DPU / w_DPU.sum()

        # ====== 3) helper per serie settimanali e score annuali ======
        import pandas as pd

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
                           m: float, lfin: float, k: float) -> dict:
            """
            Score annuale: (1-m)*(mean_r - 0.5*lfin*var_r) + m*(k * w' ESG_year)
            dove mean/var sono su rendimenti settimanali del portafoglio nell'anno.
            """
            esg_cols = esg_df.columns[: len(w)]
            esg_port_year = (esg_df[esg_cols] @ _np.asarray(w[:len(esg_cols)], float)).to_dict()
            out = {}
            if port_series is None or port_series.empty:
                return out
            df = port_series.to_frame("r")
            df["year"] = df.index.year
            grp = df.groupby("year")["r"]
            mean_r = grp.mean()
            var_r = grp.var(ddof=1)
            for y in mean_r.index:
                e_y = esg_port_year.get(pd.Timestamp(year=int(y), month=1, day=1), _np.nan)
                if _np.isnan(e_y):
                    e_y = esg_port_year.get(int(y), _np.nan)
                s_y = (1.0 - m) * (mean_r.loc[y] - 0.5 * lfin * (var_r.loc[y] if _np.isfinite(var_r.loc[y]) else 0.0)) \
                      + m * (k * (e_y if _np.isfinite(e_y) else 0.0))
                out[int(y)] = float(s_y)
            return out

        def _cum_from_simple(series: pd.Series) -> pd.Series:
            """Trasforma rendimenti semplici settimanali (%) in cumulata (%) con compounding."""
            s = series.astype(float) / 100.0
            return (1.0 + s).cumprod().sub(1.0).mul(100.0)

        def _var_cvar(port_series: pd.Series, alpha: float):
            """
            Restituisce VaR e CVaR empirici sulle perdite (-r) al livello 'alpha'.
            Ritorna None se la serie non contiene valori finiti.
            """
            if port_series is None or port_series.empty:
                return None
            vals = _np.asarray(port_series, float)
            vals = vals[_np.isfinite(vals)]
            if vals.size == 0:
                return None
            losses = -vals  # perdite in % (stesso segno dei rendimenti percentuali)
            losses = _np.sort(losses)
            T = losses.size
            k = int(_np.ceil(alpha * T))
            k = max(1, min(k, T))
            var_alpha = losses[k - 1]
            tail_sum = losses[k:].sum()
            frac = k - alpha * T
            cvar = (frac * var_alpha + tail_sum) / ((1.0 - alpha) * T)
            return {"VaR": float(var_alpha), "CVaR": float(cvar)}

        # ====== 3.bis) NEW helper: serie ESG annuale del portafoglio ======
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
        # NEW: ESG annuale (MEU/DPU) in-sample
        esg_MEU_yearly_in = esg_DPU_yearly_in = None
        # NEW: metriche di rischio (VaR/CVaR) in-sample
        risk_MEU_in = risk_DPU_in = None

        fin_df = cfg.get("fin_df", None)   # DataFrame returns settimanali della finestra
        esg_df = cfg.get("esg_df", None)   # DataFrame ESG annuale della finestra
        if fin_df is not None and esg_df is not None:
            perf_MEU_in = _portfolio_series(fin_df, w_MEU)
            perf_DPU_in = _portfolio_series(fin_df, w_DPU)
            perf_MEU_in_cum = _cum_from_simple(perf_MEU_in)
            perf_DPU_in_cum = _cum_from_simple(perf_DPU_in)
            score_MEU_yearly_in = _annual_scores(perf_MEU_in, esg_df, w_MEU, m, lfin, k_val)
            score_DPU_yearly_in = _annual_scores(perf_DPU_in, esg_df, w_DPU, m, lfin, k_val)

            # === NEW: serie ESG portafoglio annuale MEU e DPU ===
            esg_MEU_yearly_in = _esg_port_series_yearly(esg_df, w_MEU)
            esg_DPU_yearly_in = _esg_port_series_yearly(esg_df, w_DPU)

            # === NEW: VaR/CVaR portafoglio (settimanale, %) ===
            risk_MEU_in = _var_cvar(perf_MEU_in, alpha_var)
            risk_DPU_in = _var_cvar(perf_DPU_in, alpha_var)

        # ====== 5) calcoli sui 2 anni successivi (se presenti) ======
        perf_MEU_next2 = perf_DPU_next2 = perf_MEU_next2_cum = perf_DPU_next2_cum = None
        score_MEU_yearly_next2 = score_DPU_yearly_next2 = {}
        # NEW: ESG annuale (MEU/DPU) next2
        esg_MEU_yearly_next2 = esg_DPU_yearly_next2 = None
        # NEW: metriche di rischio (VaR/CVaR) next2
        risk_MEU_next2 = risk_DPU_next2 = None

        nxt = cfg.get("next2", {})
        fin_df_n2 = nxt.get("FIN_df", None)
        esg_df_n2 = nxt.get("ESG_df", None)
        if fin_df_n2 is not None and esg_df_n2 is not None:
            perf_MEU_next2 = _portfolio_series(fin_df_n2, w_MEU)
            perf_DPU_next2 = _portfolio_series(fin_df_n2, w_DPU)
            perf_MEU_next2_cum = _cum_from_simple(perf_MEU_next2)
            perf_DPU_next2_cum = _cum_from_simple(perf_DPU_next2)
            score_MEU_yearly_next2 = _annual_scores(perf_MEU_next2, esg_df_n2, w_MEU, m, lfin, k_val)
            score_DPU_yearly_next2 = _annual_scores(perf_DPU_next2, esg_df_n2, w_DPU, m, lfin, k_val)

            # === NEW: serie ESG portafoglio annuale MEU e DPU ===
            esg_MEU_yearly_next2 = _esg_port_series_yearly(esg_df_n2, w_MEU)
            esg_DPU_yearly_next2 = _esg_port_series_yearly(esg_df_n2, w_DPU)

            # === NEW: VaR/CVaR portafoglio next2 (settimanale, %) ===
            risk_MEU_next2 = _var_cvar(perf_MEU_next2, alpha_var)
            risk_DPU_next2 = _var_cvar(perf_DPU_next2, alpha_var)

        # ====== 6) return ======
        res = {
            "ok": True,
            "ws": ws, "win_tag": win_tag,
            "i_m": i_m, "i_f": i_f, "i_e": i_e,
            "m": m, "lmbd_fin": lfin, "lmbd_esg": lesg,
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

                # NEW: ESG annuale del portafoglio (MEU/DPU) in-sample
                "esg_MEU_yearly_in": esg_MEU_yearly_in,      # pd.Series (annuale)
                "esg_DPU_yearly_in": esg_DPU_yearly_in,      # pd.Series (annuale)

                # NEW: VaR/CVaR settimanale (in-sample)
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

                # NEW: ESG annuale del portafoglio (MEU/DPU) next2
                "esg_MEU_yearly_next2": esg_MEU_yearly_next2,  # pd.Series (annuale)
                "esg_DPU_yearly_next2": esg_DPU_yearly_next2,  # pd.Series (annuale)

                # NEW: VaR/CVaR settimanale (next2)
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
