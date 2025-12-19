#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 15:26:33 2025

@author: giorgio
"""

import pickle
import numpy as np
import pandas as pd
from codice_base import double_mean_variance_objective, simplex_optimize

data_ESG = pd.read_excel('sp100.xlsx', sheet_name='ESG SCORE',     skiprows=4)
data_fin = pd.read_excel('sp100.xlsx', sheet_name='WEEKLY PRICES', skiprows=4)

# Prima colonna come indice
data_ESG = data_ESG.set_index(data_ESG.columns[0])
data_fin = data_fin.set_index(data_fin.columns[0])

# Indice prezzi → datetime robusto
data_fin.index = pd.to_datetime(data_fin.index, errors="coerce")
data_fin = data_fin[~data_fin.index.isna()]
data_fin = data_fin['2008-01-01':]

# ESG: rimuovi le prime 3 righe (header/tabella) e trasforma l'indice anno→datetime
data_ESG = data_ESG.iloc[3:]
data_ESG.index = pd.to_datetime(data_ESG.index.astype(str), format='%Y', errors="coerce")
data_ESG = data_ESG[~data_ESG.index.isna()]

# ====== (IMPORTANTE) ALLINEA PER POSIZIONE, non per nome ======
MIN_ESG_COUNT = 16  # metti 0 per non filtrare
mask_cols = (data_ESG.count() >= MIN_ESG_COUNT).to_numpy()
pos_hold = np.where(mask_cols)[0]

if pos_hold.size == 0:
    print("[WARN] Nessuna colonna supera la soglia ESG; uso fallback posizionale puro.")
    pos_hold = np.arange(min(data_ESG.shape[1], data_fin.shape[1]))

max_idx = min(data_ESG.shape[1], data_fin.shape[1]) - 1
pos_hold = pos_hold[pos_hold <= max_idx]

data_ESG = data_ESG.iloc[:, pos_hold].copy()
data_ESG=data_ESG.ffill()
data_fin  = data_fin.iloc[:, pos_hold].copy()
# Elimina colonne prezzi con NaN e applica lo stesso drop a ESG
fin_to_drop = np.unique(np.where(data_fin.isna())[1])
if fin_to_drop.size > 0:
    data_fin = data_fin.drop(columns=data_fin.columns[fin_to_drop])
    data_ESG = data_ESG.drop(columns=data_ESG.columns[fin_to_drop])

def _make_psd(S: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    S = np.asarray(S, float)
    S = 0.5 * (S + S.T)
    d = np.diag(S).copy()
    bad = ~np.isfinite(d) | (d <= 0)
    if bad.any():
        good_mean = np.nanmean(d[~bad]) if (~bad).any() else 1.0
        d[bad] = good_mean if np.isfinite(good_mean) else 1.0
        d[~np.isfinite(d)] = 1.0
        S[np.diag_indices_from(S)] = d
    return S + eps * np.eye(S.shape[0], dtype=float)
returns = data_fin.pct_change().dropna() * 100
# ====== 3) STATISTICHE GLOBALI (per valutazione) ======
mu_fin_glob    = returns.mean(skipna=True).values.astype(float)
Sigma_fin_glob = _make_psd(returns.cov(min_periods=max(5, int(0.1 * len(returns)))).values.astype(float))
mu_esg_glob    = data_ESG.mean(skipna=True).values.astype(float)
Sigma_esg_glob = _make_psd(data_ESG.cov(min_periods=2).values.astype(float))
diag_esg_glob = np.clip(np.diag(Sigma_esg_glob), 1e-12, np.inf)
k_global = float(np.mean(mu_esg_glob / np.sqrt(diag_esg_glob)))
######### Mean_variance #####
with open('cv_weights_nested.pkl','rb') as e:
    data_MV=pickle.load(e)
def _rel_gap(best, value):
    """
    Restituisce il gap relativo in percentuale (best - value) / |best|.
    Se best è ~0 evita divisioni per zero.
    """
    denom = abs(best) if abs(best) > 1e-12 else 1.0
    return float((abs(best - value) / denom) * 100.0)


def build_global_eval(data):
    out = {}
    # Memoization: salva (w_star, U_max) per ciascuna combinazione (m,lfin,lesg,n)
    _cache_opt = {}

    def _solve_opt_once(m, lfin, lesg, n):
        key = (float(m), float(lfin), float(lesg), int(n))
        if key not in _cache_opt:
            # obiettivo fissando i parametri
            def U_glob(w):
                return double_mean_variance_objective(
                    w,
                    mu_fin_glob, mu_esg_glob,
                    Sigma_fin_glob, Sigma_esg_glob,
                    float(lfin), float(lesg), float(m), k_global
                )
            w_star, U_star_neg, _ = simplex_optimize(lambda w: U_glob(w), n)
            U_max = float(U_glob(w_star))
            _cache_opt[key] = (w_star, U_max)
        return _cache_opt[key]

    for ws, ws_dict in data['weights'].items():
        for win, win_dict in ws_dict.items():
            for key_tuple, r in win_dict.items():
                m    = float(r["m"])
                lfin = float(r["lmbd_fin"])
                lesg = float(r["lmbd_esg"])
                w_MEU = np.asarray(r["w_MEU"], dtype=float).reshape(-1)
                w_DPU = np.asarray(r["w_DPU"], dtype=float).reshape(-1)

                # dimensione del simplex (usa la lunghezza dei pesi)
                n = int(w_MEU.size)
                # ottimo globale calcolato/riusato dalla cache
                w_star_glob, U_max_glob = _solve_opt_once(m, lfin, lesg, n)

                # stessa utility per valutare i candidati con questi parametri
                def U_glob_eval(w):
                    return double_mean_variance_objective(
                        np.asarray(w, float),
                        mu_fin_glob, mu_esg_glob,
                        Sigma_fin_glob, Sigma_esg_glob,
                        lfin, lesg, m, k_global
                    )

                U_MEU_glob = float(U_glob_eval(w_MEU))
                U_DPU_glob = float(U_glob_eval(w_DPU))

                out.setdefault(ws, {}).setdefault(win, {})[key_tuple] = {
                    "U_max_glob": U_max_glob,
                    "U_MEU_glob": U_MEU_glob,
                    "U_DPU_glob": U_DPU_glob,
                    "gap_MEU_vs_Umax_pct": _rel_gap(U_max_glob, U_MEU_glob),
                    "gap_DPU_vs_Umax_pct": _rel_gap(U_max_glob, U_DPU_glob),
                    "gap_DPU_vs_MEU_pct": float(
                        ((U_MEU_glob - U_DPU_glob) /
                         (abs(U_MEU_glob) if abs(U_MEU_glob) > 1e-12 else 1.0)) * 100.0
                    ),
                    "w_star_glob": np.asarray(w_star_glob, float).tolist(),
                }
    return out

global_eval_mean_var = build_global_eval(data_MV)

####CVaR
with open('cv_weights_nested_CVaR.pkl','rb') as e:
    data_CVaR=pickle.load(e)
def _portfolio_series(fin_df: pd.DataFrame, w: np.ndarray) -> pd.Series:
    if fin_df is None or fin_df.empty:
        return pd.Series(dtype=float)
    W = np.asarray(w, float).reshape(-1)
    cols = fin_df.columns[: len(W)]
    r = fin_df[cols]
    def _row_dot(row_vals):
        mask = np.isfinite(row_vals)
        if not mask.any():
            return np.nan
        ww = W[:len(row_vals)][mask]
        ww = ww / ww.sum() if ww.sum() != 0 else ww
        return float(np.dot(row_vals[mask], ww))
    return r.apply(lambda row: _row_dot(row.values.astype(float)), axis=1)

def _empirical_cvar_from_losses(L, alpha: float):
    L = np.asarray(L, float)
    if L.size == 0 or not np.isfinite(L).any():
        return np.nan
    L = L[np.isfinite(L)]
    s = np.sort(L)
    T = s.size
    k = int(np.ceil(alpha * T)); k = max(1, min(k, T))
    var_alpha = s[k-1]
    tail_sum = s[k:].sum()
    frac = k - alpha * T
    return float((frac * var_alpha + tail_sum) / ((1.0 - alpha) * T))

def _U_total_global_CVaR(fin_df, mu_esg, Sigma_esg, w, lfin, lesg, m, k, alpha):
    port = _portfolio_series(fin_df, w)
    r = np.asarray(port, float) / 100.0
    L = -r
    cvar = _empirical_cvar_from_losses(L, alpha)
    mean_r = float(np.nanmean(r))
    u_fin = lfin * mean_r - (1.0 - lfin) * cvar
    u_esg = float(lesg * np.dot(w, mu_esg) - (1.0 - lesg) * np.dot(w, Sigma_esg @ w))
    return float((1.0 - m) * u_fin + m * (k * u_esg))
# ======  CVaR: valutazione globale analoga al caso mean–variance  ======


def build_global_eval_cvar(data_cvar, fin_returns_df, mu_esg_glob, Sigma_esg_glob, k_global):
    out = {}
    # cache: ottimo per (m, lfin, lesg, alpha, n)
    _cache_opt = {}

    def U_glob_cvar(w, lfin, lesg, m, alpha):
        return _U_total_global_CVaR(
            fin_df=fin_returns_df,
            mu_esg=mu_esg_glob,
            Sigma_esg=Sigma_esg_glob,
            w=np.asarray(w, float),
            lfin=float(lfin),
            lesg=float(lesg),
            m=float(m),
            k=float(k_global),
            alpha=float(alpha),
        )

    def _solve_opt_once(m, lfin, lesg, alpha, n):
        key = (float(m), float(lfin), float(lesg), float(alpha), int(n))
        if key not in _cache_opt:
            def _obj(w):
                return U_glob_cvar(w, lfin, lesg, m, alpha)
            w_star, U_star_neg, _ = simplex_optimize(lambda w: _obj(w), n)
            U_max = float(_obj(w_star))
            _cache_opt[key] = (w_star, U_max)
        return _cache_opt[key]

    for ws, ws_dict in data_cvar['weights'].items():
        for win, win_dict in ws_dict.items():
            for key_tuple, r in win_dict.items():
                # Parametri del problema
                m     = float(r.get("m", 0.5))
                lfin  = float(r.get("lmbd_fin", 0.5))
                lesg  = float(r.get("lmbd_esg", 0.5))
                alpha = float(r.get("alpha", 0.95))

                # Pesi candidati
                w_MEU = np.asarray(r.get("w_MEU", []), dtype=float).reshape(-1)
                w_DPU = np.asarray(r.get("w_DPU", []), dtype=float).reshape(-1)
                if w_MEU.size == 0 or w_DPU.size == 0:
                    continue

                # Allinea la dimensione (nel caso raro di size diverse, usa il min)
                n = int(min(w_MEU.size, w_DPU.size))
                if w_MEU.size != n: w_MEU = w_MEU[:n]
                if w_DPU.size != n: w_DPU = w_DPU[:n]

                # Ottimo globale calcolato UNA SOLA VOLTA per (m,lfin,lesg,alpha,n)
                w_star_glob, U_max_glob = _solve_opt_once(m, lfin, lesg, alpha, n)

                # Valutazioni dei candidati
                U_MEU_glob = float(U_glob_cvar(w_MEU, lfin, lesg, m, alpha))
                U_DPU_glob = float(U_glob_cvar(w_DPU, lfin, lesg, m, alpha))

                out.setdefault(ws, {}).setdefault(win, {})[key_tuple] = {
                    "alpha": alpha,
                    "U_max_glob": U_max_glob,
                    "U_MEU_glob": U_MEU_glob,
                    "U_DPU_glob": U_DPU_glob,
                    "gap_MEU_vs_Umax_pct": _rel_gap(U_max_glob, U_MEU_glob),
                    "gap_DPU_vs_Umax_pct": _rel_gap(U_max_glob, U_DPU_glob),
                    "gap_DPU_vs_MEU_pct": float(
                        ((U_MEU_glob - U_DPU_glob) /
                         (abs(U_MEU_glob) if abs(U_MEU_glob) > 1e-12 else 1.0)) * 100.0
                    ),
                    "w_star_glob": np.asarray(w_star_glob, float).tolist(),
                }
    return out

# ====== ESECUZIONE ======
global_eval_cvar = build_global_eval_cvar(
    data_CVaR,
    fin_returns_df=returns,         # usa i rendimenti % già calcolati (coerenti con _U_total_global_CVaR)
    mu_esg_glob=mu_esg_glob,
    Sigma_esg_glob=Sigma_esg_glob,
    k_global=k_global
)



def build_summary_table(global_eval_dict):
    rows = []

    # Naviga nella struttura annidata
    for ws, ws_dict in global_eval_dict.items():
        for win, win_dict in ws_dict.items():
            for key_tuple, r in win_dict.items():
                # Estraggo parametri dal key_tuple
                # supponiamo che key_tuple = (m, l_fin, l_esg) o (m, l_fin, l_esg, alpha)
                m = key_tuple[0]
                lambda_esg = key_tuple[2]  # la terza posizione è λ_ESG

                rows.append({
                    "m": m,
                    "lambda_esg": lambda_esg,
                    "gap_MEU_vs_Umax_pct": r["gap_MEU_vs_Umax_pct"],
                    "gap_DPU_vs_Umax_pct": r["gap_DPU_vs_Umax_pct"],
                })

    df = pd.DataFrame(rows)

    # Raggruppo per m e lambda_esg
    summary = df.groupby(["m", "lambda_esg"]).agg(
        MEU_gap_mean=("gap_MEU_vs_Umax_pct", "mean"),
        MEU_gap_std=("gap_MEU_vs_Umax_pct", "std"),
        MEU_gap_min=("gap_MEU_vs_Umax_pct", "min"),
        MEU_gap_max=("gap_MEU_vs_Umax_pct", "max"),

        DPU_gap_mean=("gap_DPU_vs_Umax_pct", "mean"),
        DPU_gap_std=("gap_DPU_vs_Umax_pct", "std"),
        DPU_gap_min=("gap_DPU_vs_Umax_pct", "min"),
        DPU_gap_max=("gap_DPU_vs_Umax_pct", "max"),
    ).reset_index()
    summary["MEU_std_div_mean"] = summary["MEU_gap_std"] / summary["MEU_gap_mean"]
    summary["DPU_std_div_mean"] = summary["DPU_gap_std"] / summary["DPU_gap_mean"]
    return summary

# ====== ESEMPIO SU MEAN-VARIANCE ======
summary_MV = build_summary_table(global_eval_mean_var)
print(summary_MV)

# ====== ESEMPIO SU CVaR ======
summary_CVaR = build_summary_table(global_eval_cvar)
print(summary_CVaR)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessario per attivare 3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

def plot_cv_surfaces_two(df_left, df_right,
                         title_general="Coefficient of variation (std/mean)",
                         subtitle_left="Mean-Variance",
                         subtitle_right="CVaR"):

    def compute_std_over_mean(df):
        if "MEU_std_div_mean" not in df.columns:
            df["MEU_std_div_mean"] = df["MEU_gap_std"] / df["MEU_gap_mean"]
        if "DPU_std_div_mean" not in df.columns:
            df["DPU_std_div_mean"] = df["DPU_gap_std"] / df["DPU_gap_mean"]
        return df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["m", "lambda_esg", "MEU_std_div_mean", "DPU_std_div_mean"]
        )

    df_left  = compute_std_over_mean(df_left)
    df_right = compute_std_over_mean(df_right)

    fig = plt.figure(figsize=(16, 8))

    # ---- mapping etichette X (m) ----
    x_positions = np.sort(df_left["m"].unique())
    x_labels = [0, 0.3, 0.5, 0.7, 0.9, 1]     # label che vuoi mostrare

    # ---- mapping etichette Y (lambda_esg) ----
    lambda_positions_left  = np.sort(df_left["lambda_esg"].unique())
    lambda_positions_right = np.sort(df_right["lambda_esg"].unique())

    lambda_mapping = {lambda_positions_left[i]: v for i, v in enumerate(["0.25", "0.50", "0.75"])}

    # ================= SUBPLOT 1 ===================
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.view_init(elev=10, azim=-120)

    ax1.plot_trisurf(df_left["m"], df_left["lambda_esg"], df_left["MEU_std_div_mean"], color="red", alpha=0.6)
    ax1.plot_trisurf(df_left["m"], df_left["lambda_esg"], df_left["DPU_std_div_mean"], color="blue",  alpha=0.6)

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, fontsize=14)

    ax1.set_yticks(lambda_positions_left)
    ax1.set_yticklabels([lambda_mapping[v] for v in lambda_positions_left], fontsize=14)

    ax1.set_xlabel("m", fontsize=16)
    ax1.set_ylabel(r"$\lambda_{2}$", fontsize=16)
    ax1.set_zlabel("std / mean", fontsize=16)

    ax1.text2D(0.5, 0.9, subtitle_left, transform=ax1.transAxes,
               ha="center", va="bottom", fontsize=26, fontweight="bold")

    ax1.legend(["MEU", "DPU"], loc="upper left",
               bbox_to_anchor=(0.79, 0.76), bbox_transform=ax1.transAxes,
               frameon=True, framealpha=0.9)

    # ================= SUBPLOT 2 ===================
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.view_init(elev=10, azim=-120)

    ax2.plot_trisurf(df_right["m"], df_right["lambda_esg"], df_right["MEU_std_div_mean"], color="red", alpha=0.6)
    ax2.plot_trisurf(df_right["m"], df_right["lambda_esg"], df_right["DPU_std_div_mean"], color="blue",  alpha=0.6)

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, fontsize=14)      # ✅ FIX QUI

    ax2.set_yticks(lambda_positions_right)
    ax2.set_yticklabels([lambda_mapping[v] for v in lambda_positions_right], fontsize=14)

    ax2.set_xlabel("m", fontsize=16)
    ax2.set_ylabel(r"$\lambda_{2}$", fontsize=16)
    ax2.set_zlabel("std / mean", fontsize=16)

    ax2.text2D(0.5, 0.9, subtitle_right, transform=ax2.transAxes,
               ha="center", va="bottom", fontsize=26, fontweight="bold")

    ax2.legend(["MEU", "DPU"], loc="upper left",
               bbox_to_anchor=(0.79, 0.76), bbox_transform=ax2.transAxes,
               frameon=True, framealpha=0.9)

    # Margini compatti
    fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.006, wspace=0.05)
    plt.show()
plot_cv_surfaces_two(summary_MV, summary_CVaR,
                     title_general="Confronto stabilità (std/mean)",
                     subtitle_left="Mean-Variance Model",
                     subtitle_right="CVaR Model")