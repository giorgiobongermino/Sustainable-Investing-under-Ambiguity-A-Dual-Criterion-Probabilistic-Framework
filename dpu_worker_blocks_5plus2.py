import os


def run_one_dpu_job(cfg):
    """
    Variante 5+2: calcola i pesi ottimi su 5 anni (in-sample) e poi
    ricalcola su 7 anni (5 + 2 anni successivi) per entrambi i metodi (MEU e DPU).

    Input atteso (chiavi principali in cfg):
      - Parametri: m, lmbd_fin, lmbd_esg, k_5, k_7, n_assets, rng_seed
      - Statistiche 5 anni: mu_fin_5, Sigma_fin_5, mu_esg_5, Sigma_esg_5, nu_fin_5, nu_esg_5
      - Statistiche 7 anni: mu_fin_7, Sigma_fin_7, mu_esg_7, Sigma_esg_7, nu_fin_7, nu_esg_7
      - Dati per serie/score:
          * fin_df_5 (ritorni %, base 5 anni)
          * esg_df_5 (ESG annuali, base 5 anni)
          * next2: dict con FIN_df (ritorni %) ed ESG_df per i 2 anni successivi
          * opzionale: fin_df_7, esg_df_7 (se non presenti, vengono costruiti concatenando 5 + next2)

    Ritorna un dict picklable con pesi e, se disponibili, serie performance/score per 5 e 7 anni.
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
        from numpyro import set_platform, enable_x64
        set_platform("cpu"); enable_x64()

        from codice_base import (
            expectation_DPU,
            simplex_optimize,
            double_mean_variance_objective,
        )

        # ----- helper (scala esterna ν) -----
        def mean_var_logp(x, mu, Sigma, lmbd, nu=1.0):
            u = lmbd*jnp.dot(x, mu) -(1.-lmbd) * jnp.dot(x, Sigma @ x)
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

        _py_random.seed(rng_seed)
        np.random.seed(rng_seed % (2**32 - 1))

        import numpy as _np
        # 5 anni
        mu_fin_5    = _np.asarray(cfg["mu_fin_5"], dtype=float)
        mu_esg_5    = _np.asarray(cfg["mu_esg_5"], dtype=float)
        Sigma_fin_5 = _np.asarray(cfg["Sigma_fin_5"], dtype=float)
        Sigma_esg_5 = _np.asarray(cfg["Sigma_esg_5"], dtype=float)
        nu_fin_5    = float(cfg.get("nu_fin_5", 1.0))
        nu_esg_5    = float(cfg.get("nu_esg_5", 1.0))
        alpha_var   = float(cfg.get("alpha_var", 0.95))

        # 7 anni
        mu_fin_7    = _np.asarray(cfg["mu_fin_7"], dtype=float)
        mu_esg_7    = _np.asarray(cfg["mu_esg_7"], dtype=float)
        Sigma_fin_7 = _np.asarray(cfg["Sigma_fin_7"], dtype=float)
        Sigma_esg_7 = _np.asarray(cfg["Sigma_esg_7"], dtype=float)
        nu_fin_7    = float(cfg.get("nu_fin_7", 1.0))
        nu_esg_7    = float(cfg.get("nu_esg_7", 1.0))

        # ====== 1) MEU (5 anni) ======
        def MEU5(x):
            return double_mean_variance_objective(
                x, mu_fin_5, mu_esg_5, Sigma_fin_5, Sigma_esg_5, lfin, lesg, m, k_5
            )
        w_MEU_5, _, _ = simplex_optimize(MEU5, n_assets)
        if w_MEU_5.sum() != 0:
            w_MEU_5 = w_MEU_5 / w_MEU_5.sum()

        # ====== 2) DPU (5 anni) ======
        mu_fin_5_j    = jnp.asarray(mu_fin_5)
        mu_esg_5_j    = jnp.asarray(mu_esg_5)
        Sigma_fin_5_j = jnp.asarray(Sigma_fin_5)
        Sigma_esg_5_j = jnp.asarray(Sigma_esg_5)
        nu_fin_5_j    = jnp.asarray(nu_fin_5, dtype=mu_fin_5_j.dtype)
        nu_esg_5_j    = jnp.asarray(nu_esg_5, dtype=mu_esg_5_j.dtype)

        def log_fin_5(x): return mean_var_logp(x, mu_fin_5_j, Sigma_fin_5_j, lfin, nu_fin_5_j)
        def log_esg_5(x): return mean_var_logp(x, mu_esg_5_j, Sigma_esg_5_j, lesg, nu_esg_5_j)

        import inspect as _inspect
        sig = _inspect.signature(expectation_DPU)
        if "rng_seed" in sig.parameters:
            w_DPU_5 = expectation_DPU(k_5, m, log_fin_5, log_esg_5, n_assets, rng_seed=rng_seed)
        else:
            w_DPU_5 = expectation_DPU(k_5, m, log_fin_5, log_esg_5, n_assets)
        w_DPU_5 = _np.asarray(w_DPU_5, dtype=float)
        if w_DPU_5.sum() != 0:
            w_DPU_5 = w_DPU_5 / w_DPU_5.sum()

        # ====== 3) MEU (7 anni) ======
        def MEU7(x):
            return double_mean_variance_objective(
                x, mu_fin_7, mu_esg_7, Sigma_fin_7, Sigma_esg_7, lfin, lesg, m, k_7
            )
        w_MEU_7, _, _ = simplex_optimize(MEU7, n_assets)
        if w_MEU_7.sum() != 0:
            w_MEU_7 = w_MEU_7 / w_MEU_7.sum()

        # ====== 4) DPU (7 anni) ======
        mu_fin_7_j    = jnp.asarray(mu_fin_7)
        mu_esg_7_j    = jnp.asarray(mu_esg_7)
        Sigma_fin_7_j = jnp.asarray(Sigma_fin_7)
        Sigma_esg_7_j = jnp.asarray(Sigma_esg_7)
        nu_fin_7_j    = jnp.asarray(nu_fin_7, dtype=mu_fin_7_j.dtype)
        nu_esg_7_j    = jnp.asarray(nu_esg_7, dtype=mu_esg_7_j.dtype)

        def log_fin_7(x): return mean_var_logp(x, mu_fin_7_j, Sigma_fin_7_j, lfin, nu_fin_7_j)
        def log_esg_7(x): return mean_var_logp(x, mu_esg_7_j, Sigma_esg_7_j, lesg, nu_esg_7_j)

        if "rng_seed" in sig.parameters:
            w_DPU_7 = expectation_DPU(k_7, m, log_fin_7, log_esg_7, n_assets, rng_seed=rng_seed)
        else:
            w_DPU_7 = expectation_DPU(k_7, m, log_fin_7, log_esg_7, n_assets)
        w_DPU_7 = _np.asarray(w_DPU_7, dtype=float)
        if w_DPU_7.sum() != 0:
            w_DPU_7 = w_DPU_7 / w_DPU_7.sum()

        # ====== 5) helper per serie settimanali e score annuali ======
        import pandas as pd

        def _portfolio_series(fin_df: pd.DataFrame, w: _np.ndarray) -> pd.Series:
            """Ritorna serie (%) del portafoglio con rinormalizzazione su colonne non-NaN a ogni riga."""
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
                           m: float, lfin: float, k: float) -> dict:
            """
            Score annuale: (1-m)*(mean_r - 0.5*lfin*var_r) + m*(k * w' ESG_year),
            con mean/var sui rendimenti del portafoglio nell'anno (serie in %).
            """
            out = {}
            if port_series is None or port_series.empty or esg_df is None or esg_df.empty:
                return out
            esg_cols = esg_df.columns[: len(w)]
            esg_port_year = (esg_df[esg_cols] @ _np.asarray(w[:len(esg_cols)], float)).to_dict()
            df = port_series.to_frame("r")
            df["year"] = df.index.year
            grp = df.groupby("year")["r"]
            mean_r = grp.mean()
            var_r  = grp.var(ddof=1)
            for y in mean_r.index:
                er = float(mean_r.loc[y])
                vr = float(var_r.loc[y]) if _np.isfinite(var_r.loc[y]) else 0.0
                esg_y = float(esg_port_year.get(int(y), _np.nan))
                out[int(y)] = (1.0 - m*k) * (er - 0.5 * lfin * vr) + m * (k * esg_y)
            return out

        def _cum_from_simple(port_series: pd.Series) -> pd.Series:
            if port_series is None or port_series.empty:
                import pandas as pd
                return pd.Series(dtype=float)
            return (1.0 + port_series.div(100.0)).cumprod().sub(1.0).mul(100.0)

        def _var_cvar(port_series: pd.Series, alpha: float):
            """
            VaR e CVaR empirici (perdite = -r) al livello 'alpha'.
            Restituisce None se la serie non ha valori finiti.
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
            """Serie annuale ESG del portafoglio: w' * ESG_year."""
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
        fin_df_5 = cfg.get("fin_df_5", None)
        esg_df_5 = cfg.get("esg_df_5", None)
        nxt = cfg.get("next2", {})
        fin_df_n2 = nxt.get("FIN_df", None)
        esg_df_n2 = nxt.get("ESG_df", None)

        # Costruisci 7 anni se non passato
        fin_df_7 = cfg.get("fin_df_7", None)
        esg_df_7 = cfg.get("esg_df_7", None)
        if fin_df_7 is None and fin_df_5 is not None and fin_df_n2 is not None:
            try:
                fin_df_7 = _np.concatenate([])  # placeholder per except
            except Exception:
                pass
            # concat in modo robusto (pandas)
            import pandas as pd
            fin_df_7 = pd.concat([fin_df_5, fin_df_n2], axis=0)
        if esg_df_7 is None and esg_df_5 is not None and esg_df_n2 is not None:
            import pandas as pd
            esg_df_7 = pd.concat([esg_df_5, esg_df_n2], axis=0)

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
            score_MEU5_in = _annual_scores(perf_MEU5_in, esg_df_5, w_MEU_5, m, lfin, k_5)
            score_DPU5_in = _annual_scores(perf_DPU5_in, esg_df_5, w_DPU_5, m, lfin, k_5)
            esg_MEU5_yearly_in = _esg_port_series_yearly(esg_df_5, w_MEU_5)
            esg_DPU5_yearly_in = _esg_port_series_yearly(esg_df_5, w_DPU_5)
            risk_MEU5_in = _var_cvar(perf_MEU5_in, alpha_var)
            risk_DPU5_in = _var_cvar(perf_DPU5_in, alpha_var)

        if fin_df_n2 is not None and esg_df_n2 is not None:
            perf_MEU5_next2 = _portfolio_series(fin_df_n2, w_MEU_5)
            perf_DPU5_next2 = _portfolio_series(fin_df_n2, w_DPU_5)
            perf_MEU5_next2_cum = _cum_from_simple(perf_MEU5_next2)
            perf_DPU5_next2_cum = _cum_from_simple(perf_DPU5_next2)
            score_MEU5_next2 = _annual_scores(perf_MEU5_next2, esg_df_n2, w_MEU_5, m, lfin, k_5)
            score_DPU5_next2 = _annual_scores(perf_DPU5_next2, esg_df_n2, w_DPU_5, m, lfin, k_5)
            esg_MEU5_yearly_next2 = _esg_port_series_yearly(esg_df_n2, w_MEU_5)
            esg_DPU5_yearly_next2 = _esg_port_series_yearly(esg_df_n2, w_DPU_5)
            risk_MEU5_next2 = _var_cvar(perf_MEU5_next2, alpha_var)
            risk_DPU5_next2 = _var_cvar(perf_DPU5_next2, alpha_var)

        # 7 anni (pesi 7) – usa la finestra combinata
        perf_MEU7_in = perf_DPU7_in = perf_MEU7_in_cum = perf_DPU7_in_cum = None
        score_MEU7_in = score_DPU7_in = {}
        esg_MEU7_yearly_in = esg_DPU7_yearly_in = None
        risk_MEU7_in = risk_DPU7_in = None

        if fin_df_7 is not None and esg_df_7 is not None:
            perf_MEU7_in = _portfolio_series(fin_df_7, w_MEU_7)
            perf_DPU7_in = _portfolio_series(fin_df_7, w_DPU_7)
            perf_MEU7_in_cum = _cum_from_simple(perf_MEU7_in)
            perf_DPU7_in_cum = _cum_from_simple(perf_DPU7_in)
            score_MEU7_in = _annual_scores(perf_MEU7_in, esg_df_7, w_MEU_7, m, lfin, k_7)
            score_DPU7_in = _annual_scores(perf_DPU7_in, esg_df_7, w_DPU_7, m, lfin, k_7)
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
