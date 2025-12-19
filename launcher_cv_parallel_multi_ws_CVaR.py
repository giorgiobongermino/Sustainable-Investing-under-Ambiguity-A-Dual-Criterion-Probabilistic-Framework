import os, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import pickle


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


from dpu_worker_blocks_CVaR import run_one_dpu_job  # worker (MEU + DPU)

# ===== seed base per riproducibilità =====
BASE_SEED = 12345

# ===== CVaR empirico: livello di confidenza (usato nel worker) =====
ALPHA_FIN = 0.95  # <-- nuovo parametro per CVaR empirico finanziario
# livello VaR/CVaR per reporting (usiamo lo stesso)
ALPHA_VAR = ALPHA_FIN

# ====== 1) CARICA DATI ======
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

# Se nulla passa il filtro, fallback: prendi tutte le colonne fino alla min larghezza
if pos_hold.size == 0:
    print("[WARN] Nessuna colonna supera la soglia ESG; uso fallback posizionale puro.")
    pos_hold = np.arange(min(data_ESG.shape[1], data_fin.shape[1]))

# Assicurati che gli indici siano validi per ENTRAMBI i dataframe
max_idx = min(data_ESG.shape[1], data_fin.shape[1]) - 1
pos_hold = pos_hold[pos_hold <= max_idx]

# Applica l'allineamento per POSIZIONE
data_ESG = data_ESG.iloc[:, pos_hold].copy()
data_ESG = data_ESG.fillna(method='ffill')
data_fin  = data_fin.iloc[:, pos_hold].copy()

# Elimina colonne prezzi con NaN e applica lo stesso drop a ESG
fin_to_drop = np.unique(np.where(data_fin.isna())[1])
if fin_to_drop.size > 0:
    data_fin = data_fin.drop(columns=data_fin.columns[fin_to_drop])
    data_ESG = data_ESG.drop(columns=data_ESG.columns[fin_to_drop])

# Ritorni % mensili
returns = data_fin.pct_change().dropna() * 100

print(f"[INFO] colonne selezionate per posizione: {len(pos_hold)}")
if (data_fin.shape[1] < 2) or (data_ESG.shape[1] < 2):
    print("[ERROR] Meno di 2 asset dopo l'allineamento posizionale. "
          "Controlla che i due fogli abbiano lo stesso ordinamento di colonne.")
    payload = {
        "base_seed": int(BASE_SEED),
        "weights": {},
        "assets_by_window": {},
        "param_grids": {
            "m_list": [0.0, 0.3, 0.5, 0.7, 0.9, 1.0],
            "lmbd_fin_list": [0.50],
            "lesg_factors": [0.5, 1.0, 2.0],
            "alpha_fin": ALPHA_FIN,  # nuovo: includo nel payload
        },
        "errors": [{"msg": "meno di 2 colonne dopo allineamento posizionale"}],
    }
    with open("cv_weights_nested_CVaR.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    raise SystemExit(0)

# ====== 2) PARAM GRID ======
m_list        = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
lmbd_fin_list = [0.5]
lesg_factors  = [0.75, 1.0, 1.5]
window_sizes  = [5, 6, 8, 10]
# Se vuoi fare grid anche su alpha_fin, crea ad es. alpha_fin_list = [0.90, 0.95, 0.99]
# e cicla sotto; per ora usiamo ALPHA_FIN fisso.
def to_host_numpy(x, dtype=float):
    try:
        import jax
        # Se è un array JAX/Tracer, completa e porta su host
        if hasattr(x, "block_until_ready"):
            x = x.block_until_ready()
        x = jax.device_get(x)
    except Exception:
        pass  # se non è JAX o è già host
    try:
        import numpy as _np
        return _np.asarray(x, dtype=dtype)
    except Exception:
        return x
# ====== util: seed deterministico per job ======
def make_job_seed(ws, sy, ey, i_m, i_f, i_e, base_seed=BASE_SEED):
    ss = np.random.SeedSequence([int(base_seed), int(ws), int(sy), int(ey), int(i_m), int(i_f), int(i_e)])
    return int(np.random.default_rng(ss).integers(0, 2**31 - 1))

# ====== util: rendi PSD e ben condizionata una covarianza ======
def _make_psd(S: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    S = np.asarray(S, float)
    S = 0.5 * (S + S.T)  # simmetrizza
    d = np.diag(S).copy()
    bad = ~np.isfinite(d) | (d <= 0)
    if bad.any():
        good_mean = np.nanmean(d[~bad]) if (~bad).any() else 1.0
        d[bad] = good_mean if np.isfinite(good_mean) else 1.0
        d[~np.isfinite(d)] = 1.0
        S[np.diag_indices_from(S)] = d
    return S + eps * np.eye(S.shape[0], dtype=float)

# ====== 3) COSTRUISCI FINESTRE & STATISTICHE ======
def build_window_stats_for_ws(ws: int):
    years_esg = data_ESG.index.year.unique()
    years_fin = returns.index.year.unique()
    if len(years_esg) == 0 or len(years_fin) == 0:
        print(f"[WARN] Nessun anno in uno degli indici (ws={ws}).")
        return []

    start_min = int(max(years_esg.min(), years_fin.min()))
    end_max   = int(min(years_esg.max(), years_fin.max()))
    if end_max - start_min + 1 < ws:
        print(f"[INFO] Orizzonte temporale insufficiente per ws={ws}.")
        return []

    start_years = list(range(start_min, end_max - ws + 2))
    out = []

    for start_year in start_years:
        end_year = start_year + ws            # esclusivo
        sy, ey = start_year, end_year - 1     # inclusivo

        win_tag = f"ws{ws}_win_{sy}_{ey}"

        # --- blocchi della finestra corrente
        mask_ann  = (data_ESG.index.year >= start_year) & (data_ESG.index.year < end_year)
        mask_week = (returns.index.year  >= start_year) & (returns.index.year  < end_year)
        block_ESG = data_ESG.loc[mask_ann]
        block_fin = returns.loc[mask_week]

        if block_ESG.empty or block_fin.empty:
            print(f"[{win_tag}] skip: uno dei due blocchi è vuoto.")
            continue
        if block_ESG.shape[1] < 2 or block_fin.shape[1] < 2:
            print(f"[{win_tag}] skip: meno di 2 asset dopo il taglio finestra.")
            continue

        # --- statistiche finestra corrente
        min_obs = max(5, int(0.1 * len(block_fin)))
        mu_fin    = block_fin.mean(skipna=True)
        Sigma_fin = block_fin.cov(min_periods=min_obs)

        mu_esg    = block_ESG.mean(skipna=True)
        Sigma_esg = block_ESG.cov(min_periods=2)

        # --- dataset numerici per uso JAX (fin_df con maschere NaN)
        fin_np_raw = block_fin.to_numpy(dtype=float, copy=True)
        fin_mask   = np.isfinite(fin_np_raw).astype(np.bool_)
        fin_np     = np.where(fin_mask, fin_np_raw, 0.0).astype(float)

        if Sigma_fin.isna().values.any() or mu_fin.isna().all():
            print(f"[{win_tag}] skip: Sigma_fin invalida o mu_fin tutto NaN.")
            continue
        if Sigma_esg.isna().values.any() or mu_esg.isna().all():
            print(f"[{win_tag}] skip: Sigma_esg invalida o mu_esg tutto NaN.")
            continue

        mu_fin_np    = mu_fin.values.astype(float)
        Sigma_fin_np = _make_psd(Sigma_fin.values.astype(float), eps=1e-8)
        mu_esg_np    = mu_esg.values.astype(float)
        Sigma_esg_np = _make_psd(Sigma_esg.values.astype(float), eps=1e-8)

        n_assets = mu_fin_np.shape[0]
        if n_assets < 2:
            print(f"[{win_tag}] skip: n_assets < 2 dopo conversione.")
            continue

        diag_esg_clip = np.clip(np.diag(Sigma_esg_np), 1e-12, np.inf)
        k_val = float(np.mean(mu_esg_np / np.sqrt(diag_esg_clip)))

        diag_fin_clip = np.clip(np.diag(Sigma_fin_np), 1e-12, np.inf)
        snr_fin  = float(np.mean(np.abs(mu_fin_np) / np.sqrt(diag_fin_clip)))
        snr_esg  = float(np.mean(np.abs(mu_esg_np) / np.sqrt(diag_esg_clip)))
        nu_fin   = float(np.sqrt(block_fin.shape[0] * snr_fin))
        nu_esg   = float(np.sqrt(block_ESG.shape[0] * snr_esg))

        # === blocchi dei 2 anni successivi alla finestra corrente ===
        # intervallo [end_year, end_year+2) ossia anni end_year e end_year+1
        mask_ann_next2  = (data_ESG.index.year >= end_year) & (data_ESG.index.year < end_year + 2)
        mask_week_next2 = (returns.index.year  >= end_year) & (returns.index.year  < end_year + 2)
        block_ESG_next2 = data_ESG.loc[mask_ann_next2]
        block_fin_next2 = returns.loc[mask_week_next2]

        next2_payload = {
            "start_year": int(end_year),
            "end_year": int(end_year + 1),  # inclusivo nel senso "ultimi due anni: end_year e end_year+1"
            "ESG_df": block_ESG_next2.copy(),   # DataFrame (pickle-able)
            "FIN_df": block_fin_next2.copy(),   # DataFrame (returns %)
        }
        if not block_fin_next2.empty and not block_ESG_next2.empty:
            mu_fin_n2    = block_fin_next2.mean(skipna=True)
            Sigma_fin_n2 = block_fin_next2.cov(min_periods=max(5, int(0.1 * len(block_fin_next2))))
            mu_esg_n2    = block_ESG_next2.mean(skipna=True)
            Sigma_esg_n2 = block_ESG_next2.cov(min_periods=2)

            fin_np_n2_raw = block_fin_next2.to_numpy(dtype=float, copy=True)
            fin_mask_n2   = np.isfinite(fin_np_n2_raw).astype(np.bool_)
            fin_np_n2     = np.where(fin_mask_n2, fin_np_n2_raw, 0.0).astype(float)

            if not (mu_fin_n2.isna().all() or mu_esg_n2.isna().all()):
                next2_payload.update({
                    "mu_fin_np":    mu_fin_n2.values.astype(float),
                    "Sigma_fin_np": _make_psd(Sigma_fin_n2.values.astype(float), eps=1e-8) if Sigma_fin_n2.size else None,
                    "mu_esg_np":    mu_esg_n2.values.astype(float),
                    "Sigma_esg_np": _make_psd(Sigma_esg_n2.values.astype(float), eps=1e-8) if Sigma_esg_n2.size else None,
                    "fin_np":       fin_np_n2,
                    "fin_mask":     fin_mask_n2,
                })

        # === dizionario finale per la finestra ===
        out.append({
            "ws": ws,
            "win_tag": win_tag,
            "start_year": sy,
            "end_year": ey,
            "mu_fin_np": mu_fin_np,
            "Sigma_fin_np": Sigma_fin_np,
            "mu_esg_np": mu_esg_np,
            "Sigma_esg_np": Sigma_esg_np,
            "n_assets": int(n_assets),
            "k": k_val,
            "nu_fin": nu_fin,
            "nu_esg": nu_esg,
            "asset_positions": list(range(n_assets)),

            # dataset numerici per componenti JAX
            "fin_np": fin_np,
            "fin_mask": fin_mask,

            # dataset per serie e score in-sample (usati per CVaR empirico fin nel worker)
            "fin_df": block_fin.copy(),
            "esg_df": block_ESG.copy(),

            # dataset per i 2 anni successivi
            "next2": next2_payload,
        })
    return out

# Costruisci finestre per tutte le ws richieste
all_window_stats = []
for ws in window_sizes:
    stats_ws = build_window_stats_for_ws(ws)
    print(f"[INFO] ws={ws}: finestre utilizzabili = {len(stats_ws)}")
    all_window_stats.extend(stats_ws)

print(f"[INFO] totale finestre utilizzabili: {len(all_window_stats)}")

# ====== 4) PREPARA JOBS ======
jobs = []
for w in all_window_stats:
    ws      = w["ws"]
    sy, ey  = w["start_year"], w["end_year"]
    for i_m, m in enumerate(m_list):
        for i_f, lfin in enumerate(lmbd_fin_list):
            for i_e, fac in enumerate(lesg_factors):
                lesg = fac * lfin
                seed = make_job_seed(ws, sy, ey, i_m, i_f, i_e, base_seed=BASE_SEED)

                cfg = {
                    "ws": ws,
                    "win_tag": w["win_tag"],
                    "i_m": i_m, "i_f": i_f, "i_e": i_e,
                    "m": float(m),
                    "lmbd_fin": float(lfin),
                    "lmbd_esg": float(lesg),
                    "k": float(w["k"]),
                    "n_assets": int(w["n_assets"]),
                    "mu_fin_np": w["mu_fin_np"],
                    "mu_esg_np": w["mu_esg_np"],
                    "Sigma_fin_np": w["Sigma_fin_np"],
                    "Sigma_esg_np": w["Sigma_esg_np"],
                    "nu_fin": float(w["nu_fin"]),
                    "nu_esg": float(w["nu_esg"]),
                    "alpha_var": float(ALPHA_VAR),
                    "rng_seed": int(seed),
                    # nuovo: livello di confidenza per CVaR empirico finanziario
                    "alpha_fin": float(ALPHA_FIN),
                }

                # passa i dataset per serie e score
                cfg["fin_np"]   = w["fin_np"]
                cfg["fin_mask"] = w["fin_mask"]
                cfg["fin_df"] = w["fin_df"]
                cfg["esg_df"] = w["esg_df"]
                cfg["next2"]  = w["next2"]

                jobs.append(cfg)

print(f"[INFO] job totali: {len(jobs)}")

# Se non ci sono job, salva payload vuoto e termina
if __name__ == "__main__" and not jobs:
    payload = {
        "base_seed": int(BASE_SEED),
        "weights": {},
        "risk_by_window": {},
        "assets_by_window": {},
        "param_grids": {
            "m_list": m_list,
            "lmbd_fin_list": lmbd_fin_list,
            "lesg_factors": lesg_factors,
            "alpha_fin": ALPHA_FIN,  # nuovo
        },
        "errors": [{"msg": "no jobs generated"}],
    }
    with open("cv_weights_nested_CVaR.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("[INFO] Nessun job generato: salvato payload vuoto 'cv_weights_nested.pkl'.")
    raise SystemExit(0)

# ====== 5) ESECUZIONE PARALLELA ======
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ctx = multiprocessing.get_context("spawn")

    N_CORES = os.cpu_count() or 8
    N_PROCESSES = min(N_CORES, len( jobs))
    if N_PROCESSES < 1:
        N_PROCESSES = 1

    print(f"[INFO] lancio con max_workers={N_PROCESSES} su {N_CORES} core")

    results = []
    with ProcessPoolExecutor(max_workers=N_PROCESSES, mp_context=ctx) as ex:
        futs = [ex.submit(run_one_dpu_job, cfg) for cfg in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())

    ok  = [r for r in results if r.get("ok")]
    err = [r for r in results if not r.get("ok")]
    if err:
        print(f"[WARN] job falliti: {len(err)} (inclusi nel payload per debug)")

    # ====== 6) DIZIONARI PICKLABLE ======
    weights         = {}  # ws -> win_tag -> (i_m,i_f,i_e) -> {...}
    perf_by_window  = {}  # ws -> win_tag -> (i_m,i_f,i_e) -> serie performance
    scores_by_window= {}  # ws -> win_tag -> (i_m,i_f,i_e) -> score annuali
    esg_by_window   = {}  # ws -> win_tag -> (i_m,i_f,i_e) -> serie ESG annuali (MEU/DPU) in + next2
    risk_by_window  = {}  # ws -> win_tag -> (i_m,i_f,i_e) -> VaR/CVaR

    for r in ok:
        ws, win = int(r["ws"]), r["win_tag"]
        key_tuple = (int(r["i_m"]), int(r["i_f"]), int(r["i_e"]))

        # --- pesi
        weights.setdefault(ws, {}).setdefault(win, {})[key_tuple] = {
            "m": float(r["m"]),
            "lmbd_fin": float(r["lmbd_fin"]),
            "lmbd_esg": float(r["lmbd_esg"]),
            "alpha_fin": float(r.get("alpha_fin", ALPHA_FIN)),  # nuovo: salvo anche alpha_fin
            "alpha_var": float(r.get("alpha_var", ALPHA_VAR)),
            "w_MEU": to_host_numpy(r["w_MEU"]),
            "w_DPU": to_host_numpy(r["w_DPU"]),
            "rng_seed": int(r.get("rng_seed", BASE_SEED)),
        }

        # --- serie performance (se presenti)
        perf_payload = {}
        if r.get("perf_MEU_in") is not None:
            perf_payload.update({
                "perf_MEU_in":        r.get("perf_MEU_in"),
                "perf_DPU_in":        r.get("perf_DPU_in"),
                "perf_MEU_in_cum":    r.get("perf_MEU_in_cum"),
                "perf_DPU_in_cum":    r.get("perf_DPU_in_cum"),
            })
        if r.get("perf_MEU_next2") is not None:
            perf_payload.update({
                "perf_MEU_next2":     r.get("perf_MEU_next2"),
                "perf_DPU_next2":     r.get("perf_DPU_next2"),
                "perf_MEU_next2_cum": r.get("perf_MEU_next2_cum"),
                "perf_DPU_next2_cum": r.get("perf_DPU_next2_cum"),
            })
        if perf_payload:
            perf_by_window.setdefault(ws, {}).setdefault(win, {})[key_tuple] = perf_payload

        # --- score annuali (se presenti)
        score_payload = {}
        if r.get("score_MEU_yearly_in") is not None:
            score_payload.update({
                "score_MEU_yearly_in": r.get("score_MEU_yearly_in"),  # dict {year:int -> score:float}
                "score_DPU_yearly_in": r.get("score_DPU_yearly_in"),
            })
        if r.get("score_MEU_yearly_next2") is not None:
            score_payload.update({
                "score_MEU_yearly_next2": r.get("score_MEU_yearly_next2"),
                "score_DPU_yearly_next2": r.get("score_DPU_yearly_next2"),
            })
        if score_payload:
            scores_by_window.setdefault(ws, {}).setdefault(win, {})[key_tuple] = score_payload

        # --- ESG annuali (MEU/DPU) in-sample e next2 (se presenti)
        esg_payload = {}
        if r.get("esg_MEU_yearly_in") is not None:
            esg_payload["esg_MEU_yearly_in"] = r.get("esg_MEU_yearly_in")
        if r.get("esg_DPU_yearly_in") is not None:
            esg_payload["esg_DPU_yearly_in"] = r.get("esg_DPU_yearly_in")
        if r.get("esg_MEU_yearly_next2") is not None:
            esg_payload["esg_MEU_yearly_next2"] = r.get("esg_MEU_yearly_next2")
        if r.get("esg_DPU_yearly_next2") is not None:
            esg_payload["esg_DPU_yearly_next2"] = r.get("esg_DPU_yearly_next2")

        if esg_payload:
            esg_by_window.setdefault(ws, {}).setdefault(win, {})[key_tuple] = esg_payload

        # --- VaR/CVaR settimanale (in-sample / next2)
        risk_payload = {}
        if r.get("risk_MEU_in") is not None:
            risk_payload["risk_MEU_in"] = r.get("risk_MEU_in")
        if r.get("risk_DPU_in") is not None:
            risk_payload["risk_DPU_in"] = r.get("risk_DPU_in")
        if r.get("risk_MEU_next2") is not None:
            risk_payload["risk_MEU_next2"] = r.get("risk_MEU_next2")
        if r.get("risk_DPU_next2") is not None:
            risk_payload["risk_DPU_next2"] = r.get("risk_DPU_next2")

        if risk_payload:
            risk_by_window.setdefault(ws, {}).setdefault(win, {})[key_tuple] = risk_payload

    # mappa posizioni colonna per ciascuna finestra (0..n-1)
    assets_by_window = {}
    for w in all_window_stats:
        assets_by_window.setdefault(w["ws"], {})[w["win_tag"]] = w["asset_positions"]

    payload = {
        "base_seed": int(BASE_SEED),
        "weights": weights,
        "perf_by_window": perf_by_window,
        "scores_by_window": scores_by_window,
        "esg_by_window": esg_by_window,          # <- contenuto ESG annuale
        "risk_by_window": risk_by_window,
        "assets_by_window": assets_by_window,
        "param_grids": {
            "m_list": m_list,
            "lmbd_fin_list": lmbd_fin_list,
            "lesg_factors": lesg_factors,
            "alpha_fin": ALPHA_FIN,              # nuovo: riportato nel payload
        },
        "errors": err,
    }

    with open("cv_weights_nested_CVaR.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Salvato: cv_weights_nested.pkl")
