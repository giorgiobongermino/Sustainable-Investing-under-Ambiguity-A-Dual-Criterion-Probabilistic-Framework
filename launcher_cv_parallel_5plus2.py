import os, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import pickle

from dpu_worker_blocks_5plus2 import run_one_dpu_job  # nuovo worker 5+2 (MEU + DPU)

# ===== seed base per riproducibilità =====
BASE_SEED = 12345
# livello per VaR/CVaR di reporting
ALPHA_VAR = 0.95

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

# Ritorni % (mensili)
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
            "lmbd_fin_list": [0.5],
            "lesg_factors": [0.75, 1.0, 1.5],
        },
        "errors": [{"msg": "meno di 2 colonne dopo allineamento posizionale"}],
    }
    with open("cv_weights_5plus2.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    raise SystemExit(0)

# ====== 2) PARAM GRID ======
m_list        = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
lmbd_fin_list = [0.5]
lesg_factors  = [0.75, 1.0, 1.5]

# ====== util: seed deterministico per job ======
def make_job_seed(sy, ey5, ey7, i_m, i_f, i_e, base_seed=BASE_SEED):
    ss = np.random.SeedSequence([int(base_seed), int(sy), int(ey5), int(ey7), int(i_m), int(i_f), int(i_e)])
    return int(np.random.default_rng(ss).integers(0, 2**31 - 1))

# ====== util: rendi PSD e ben condizionata una covarianza ======
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

# ====== 3) COSTRUISCI FINESTRE 5+2 & STATISTICHE ======
def build_5plus2_windows():
    years_esg = data_ESG.index.year.unique()
    years_fin = returns.index.year.unique()
    if len(years_esg) == 0 or len(years_fin) == 0:
        print(f"[WARN] Nessun anno disponibile per costruire le finestre 5+2.")
        return []

    start_min = int(max(years_esg.min(), years_fin.min()))
    end_max   = int(min(years_esg.max(), years_fin.max()))
    ws5 = 5
    # servono almeno 7 anni totali per avere 5+2
    if end_max - start_min + 1 < 7:
        print("[INFO] Orizzonte temporale insufficiente per finestre 5+2.")
        return []

    start_years = list(range(start_min, end_max - 7 + 2))  # ultimo start tale che [start, start+7) ⊆ range
    out = []

    for start_year in start_years:
        end5 = start_year + 5      # esclusivo
        end7 = start_year + 7      # esclusivo (5 + 2)
        sy, ey5 = start_year, end5 - 1
        ey7 = end7 - 1

        win_tag = f"ws5plus2_win_{sy}_{ey5}_to_{ey7}"

        # --- blocchi 5 anni ---
        mask_ann5  = (data_ESG.index.year >= start_year) & (data_ESG.index.year < end5)
        mask_ret5  = (returns.index.year  >= start_year) & (returns.index.year  < end5)
        ESG_5 = data_ESG.loc[mask_ann5]
        FIN_5 = returns.loc[mask_ret5]

        # --- blocchi 2 anni successivi ---
        mask_ann2  = (data_ESG.index.year >= end5) & (data_ESG.index.year < end7)
        mask_ret2  = (returns.index.year  >= end5) & (returns.index.year  < end7)
        ESG_2 = data_ESG.loc[mask_ann2]
        FIN_2 = returns.loc[mask_ret2]

        if ESG_5.empty or FIN_5.empty:
            print(f"[{win_tag}] skip: blocchi 5 anni vuoti.")
            continue
        if ESG_5.shape[1] < 2 or FIN_5.shape[1] < 2:
            print(f"[{win_tag}] skip: meno di 2 asset dopo il taglio 5 anni.")
            continue
        if ESG_2.empty or FIN_2.empty:
            print(f"[{win_tag}] skip: non disponibili i 2 anni successivi (necessari per 7 anni).")
            continue

        # --- 7 anni (concat di 5 + 2) ---
        ESG_7 = pd.concat([ESG_5, ESG_2], axis=0)
        FIN_7 = pd.concat([FIN_5, FIN_2], axis=0)

        # --- statistiche: 5 anni ---
        min_obs5 = max(5, int(0.1 * len(FIN_5)))
        mu_fin_5    = FIN_5.mean(skipna=True)
        Sigma_fin_5 = FIN_5.cov(min_periods=min_obs5)
        mu_esg_5    = ESG_5.mean(skipna=True)
        Sigma_esg_5 = ESG_5.cov(min_periods=2)

        if Sigma_fin_5.isna().values.any() or mu_fin_5.isna().all():
            print(f"[{win_tag}] skip: Sigma_fin_5 invalida o mu_fin_5 tutto NaN.")
            continue
        if Sigma_esg_5.isna().values.any() or mu_esg_5.isna().all():
            print(f"[{win_tag}] skip: Sigma_esg_5 invalida o mu_esg_5 tutto NaN.")
            continue

        mu_fin_5_np    = mu_fin_5.values.astype(float)
        Sigma_fin_5_np = _make_psd(Sigma_fin_5.values.astype(float), eps=1e-8)
        mu_esg_5_np    = mu_esg_5.values.astype(float)
        Sigma_esg_5_np = _make_psd(Sigma_esg_5.values.astype(float), eps=1e-8)

        # --- statistiche: 7 anni ---
        min_obs7 = max(5, int(0.1 * len(FIN_7)))
        mu_fin_7    = FIN_7.mean(skipna=True)
        Sigma_fin_7 = FIN_7.cov(min_periods=min_obs7)
        mu_esg_7    = ESG_7.mean(skipna=True)
        Sigma_esg_7 = ESG_7.cov(min_periods=2)

        if Sigma_fin_7.isna().values.any() or mu_fin_7.isna().all():
            print(f"[{win_tag}] skip: Sigma_fin_7 invalida o mu_fin_7 tutto NaN.")
            continue
        if Sigma_esg_7.isna().values.any() or mu_esg_7.isna().all():
            print(f"[{win_tag}] skip: Sigma_esg_7 invalida o mu_esg_7 tutto NaN.")
            continue

        mu_fin_7_np    = mu_fin_7.values.astype(float)
        Sigma_fin_7_np = _make_psd(Sigma_fin_7.values.astype(float), eps=1e-8)
        mu_esg_7_np    = mu_esg_7.values.astype(float)
        Sigma_esg_7_np = _make_psd(Sigma_esg_7.values.astype(float), eps=1e-8)

        # --- parametri k e nu per 5 e 7 anni ---
        diag_esg5_clip = np.clip(np.diag(Sigma_esg_5_np), 1e-12, np.inf)
        k_5 = float(np.mean(mu_esg_5_np / np.sqrt(diag_esg5_clip)))
        diag_esg7_clip = np.clip(np.diag(Sigma_esg_7_np), 1e-12, np.inf)
        k_7 = float(np.mean(mu_esg_7_np / np.sqrt(diag_esg7_clip)))

        diag_fin5_clip = np.clip(np.diag(Sigma_fin_5_np), 1e-12, np.inf)
        snr_fin_5 = float(np.mean(np.abs(mu_fin_5_np) / np.sqrt(diag_fin5_clip)))
        diag_fin7_clip = np.clip(np.diag(Sigma_fin_7_np), 1e-12, np.inf)
        snr_fin_7 = float(np.mean(np.abs(mu_fin_7_np) / np.sqrt(diag_fin7_clip)))

        snr_esg_5 = float(np.mean(np.abs(mu_esg_5_np) / np.sqrt(diag_esg5_clip)))
        snr_esg_7 = float(np.mean(np.abs(mu_esg_7_np) / np.sqrt(diag_esg7_clip)))

        nu_fin_5 = float(np.sqrt(FIN_5.shape[0] * snr_fin_5))
        nu_esg_5 = float(np.sqrt(ESG_5.shape[0] * snr_esg_5))
        nu_fin_7 = float(np.sqrt(FIN_7.shape[0] * snr_fin_7))
        nu_esg_7 = float(np.sqrt(ESG_7.shape[0] * snr_esg_7))

        out.append({
            "win_tag": win_tag,
            "start_year": int(sy),
            "end_year_5": int(ey5),
            "end_year_7": int(ey7),

            # 5 anni
            "mu_fin_5": mu_fin_5_np,
            "Sigma_fin_5": Sigma_fin_5_np,
            "mu_esg_5": mu_esg_5_np,
            "Sigma_esg_5": Sigma_esg_5_np,
            "nu_fin_5": float(nu_fin_5),
            "nu_esg_5": float(nu_esg_5),

            # 7 anni
            "mu_fin_7": mu_fin_7_np,
            "Sigma_fin_7": Sigma_fin_7_np,
            "mu_esg_7": mu_esg_7_np,
            "Sigma_esg_7": Sigma_esg_7_np,
            "nu_fin_7": float(nu_fin_7),
            "nu_esg_7": float(nu_esg_7),

            # dimensione
            "n_assets": int(mu_fin_5_np.shape[0]),

            # dataset per serie e score
            "fin_df_5": FIN_5.copy(),
            "esg_df_5": ESG_5.copy(),
            "next2": {"FIN_df": FIN_2.copy(), "ESG_df": ESG_2.copy()},
            "fin_df_7": FIN_7.copy(),
            "esg_df_7": ESG_7.copy(),

            # fiducia ESG (k) differenziata per finestra
            "k_5": k_5,
            "k_7": k_7,
        })

    return out


window_stats = build_5plus2_windows()
print(f"[INFO] finestre 5+2 costruite: {len(window_stats)}")

# ====== 4) PREPARA JOBS ======
jobs = []
for w in window_stats:
    sy, ey5, ey7 = w["start_year"], w["end_year_5"], w["end_year_7"]
    for i_m, m in enumerate(m_list):
        for i_f, lfin in enumerate(lmbd_fin_list):
            for i_e, fac in enumerate(lesg_factors):
                lesg = fac * lfin
                seed = make_job_seed(sy, ey5, ey7, i_m, i_f, i_e, base_seed=BASE_SEED)

                cfg = {
                    "win_tag": w["win_tag"],
                    "i_m": i_m, "i_f": i_f, "i_e": i_e,
                    "m": float(m),
                    "lmbd_fin": float(lfin),
                    "lmbd_esg": float(lesg),
                    "n_assets": int(w["n_assets"]),
                    "rng_seed": int(seed),

                    # k per 5 e 7 anni
                    "k_5": float(w["k_5"]),
                    "k_7": float(w["k_7"]),

                    # 5 anni
                    "mu_fin_5": w["mu_fin_5"],
                    "mu_esg_5": w["mu_esg_5"],
                    "Sigma_fin_5": w["Sigma_fin_5"],
                    "Sigma_esg_5": w["Sigma_esg_5"],
                    "nu_fin_5": float(w["nu_fin_5"]),
                    "nu_esg_5": float(w["nu_esg_5"]),

                    # 7 anni
                    "mu_fin_7": w["mu_fin_7"],
                    "mu_esg_7": w["mu_esg_7"],
                    "Sigma_fin_7": w["Sigma_fin_7"],
                    "Sigma_esg_7": w["Sigma_esg_7"],
                    "nu_fin_7": float(w["nu_fin_7"]),
                    "nu_esg_7": float(w["nu_esg_7"]),

                    # dataset per serie/score
                    "fin_df_5": w["fin_df_5"],
                    "esg_df_5": w["esg_df_5"],
                    "next2": w["next2"],
                    "fin_df_7": w["fin_df_7"],
                    "esg_df_7": w["esg_df_7"],
                    "alpha_var": float(ALPHA_VAR),
                }

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
        },
        "errors": [{"msg": "no jobs generated"}],
    }
    with open("cv_weights_5plus2.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("[INFO] Nessun job generato: salvato payload vuoto 'cv_weights_5plus2.pkl'.")
    raise SystemExit(0)

# ====== 5) ESECUZIONE PARALLELA ======
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ctx = multiprocessing.get_context("spawn")

    N_CORES = os.cpu_count() or 8
    N_PROCESSES = min(N_CORES, len(jobs))
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
    weights         = {}  # win_tag -> (i_m,i_f,i_e) -> pesi 5 e 7
    perf_by_window  = {}  # win_tag -> (i_m,i_f,i_e) -> {perf_5: {...}, perf_7: {...}}
    scores_by_window= {}  # win_tag -> (i_m,i_f,i_e) -> {scores_5: {...}, scores_7: {...}}
    esg_by_window   = {}  # win_tag -> (i_m,i_f,i_e) -> {esg_5: {...}, esg_7: {...}}
    risk_by_window  = {}  # win_tag -> (i_m,i_f,i_e) -> {risk_5: {...}, risk_7: {...}}

    for r in ok:
        win = r["win_tag"]
        key_tuple = (int(r["i_m"]), int(r["i_f"]), int(r["i_e"]))

        # --- pesi
        weights.setdefault(win, {})[key_tuple] = {
            "m": float(r["m"]),
            "lmbd_fin": float(r["lmbd_fin"]),
            "lmbd_esg": float(r["lmbd_esg"]),
            "alpha_var": float(r.get("alpha_var", ALPHA_VAR)),
            "k_5": float(r["k_5"]),
            "k_7": float(r["k_7"]),
            "w_MEU_5": np.asarray(r["w_MEU_5"]),
            "w_DPU_5": np.asarray(r["w_DPU_5"]),
            "w_MEU_7": np.asarray(r["w_MEU_7"]),
            "w_DPU_7": np.asarray(r["w_DPU_7"]),
            "rng_seed": int(r.get("rng_seed", BASE_SEED)),
        }

        # --- serie performance (se presenti)
        perf_payload_5 = {}
        if r.get("perf_MEU5_in") is not None:
            perf_payload_5.update({
                "perf_MEU_in":        r.get("perf_MEU5_in"),
                "perf_DPU_in":        r.get("perf_DPU5_in"),
                "perf_MEU_in_cum":    r.get("perf_MEU5_in_cum"),
                "perf_DPU_in_cum":    r.get("perf_DPU5_in_cum"),
            })
        if r.get("perf_MEU5_next2") is not None:
            perf_payload_5.update({
                "perf_MEU_next2":     r.get("perf_MEU5_next2"),
                "perf_DPU_next2":     r.get("perf_DPU5_next2"),
                "perf_MEU_next2_cum": r.get("perf_MEU5_next2_cum"),
                "perf_DPU_next2_cum": r.get("perf_DPU5_next2_cum"),
            })

        perf_payload_7 = {}
        if r.get("perf_MEU7_in") is not None:
            perf_payload_7.update({
                "perf_MEU_in":        r.get("perf_MEU7_in"),
                "perf_DPU_in":        r.get("perf_DPU7_in"),
                "perf_MEU_in_cum":    r.get("perf_MEU7_in_cum"),
                "perf_DPU_in_cum":    r.get("perf_DPU7_in_cum"),
            })

        perf_by_window.setdefault(win, {})[key_tuple] = {
            "perf_5": perf_payload_5,
            "perf_7": perf_payload_7,
        }

        # --- score annuali (se presenti)
        scores_5 = {}
        if r.get("score_MEU5_in") is not None:
            scores_5.update({
                "score_MEU_in": r.get("score_MEU5_in"),
                "score_DPU_in": r.get("score_DPU5_in"),
            })
        if r.get("score_MEU5_next2") is not None:
            scores_5.update({
                "score_MEU_next2": r.get("score_MEU5_next2"),
                "score_DPU_next2": r.get("score_DPU5_next2"),
            })

        scores_7 = {}
        if r.get("score_MEU7_in") is not None:
            scores_7.update({
                "score_MEU_in": r.get("score_MEU7_in"),
                "score_DPU_in": r.get("score_DPU7_in"),
            })

        scores_by_window.setdefault(win, {})[key_tuple] = {
            "scores_5": scores_5,
            "scores_7": scores_7,
        }

        # --- ESG annuali (MEU/DPU) in-sample e next2 (5 anni) + in-sample 7 anni
        esg_5 = {}
        if r.get("esg_MEU5_yearly_in") is not None:
            esg_5["esg_MEU_yearly_in"] = r.get("esg_MEU5_yearly_in")
        if r.get("esg_DPU5_yearly_in") is not None:
            esg_5["esg_DPU_yearly_in"] = r.get("esg_DPU5_yearly_in")
        if r.get("esg_MEU5_yearly_next2") is not None:
            esg_5["esg_MEU_yearly_next2"] = r.get("esg_MEU5_yearly_next2")
        if r.get("esg_DPU5_yearly_next2") is not None:
            esg_5["esg_DPU_yearly_next2"] = r.get("esg_DPU5_yearly_next2")

        esg_7 = {}
        if r.get("esg_MEU7_yearly_in") is not None:
            esg_7["esg_MEU_yearly_in"] = r.get("esg_MEU7_yearly_in")
        if r.get("esg_DPU7_yearly_in") is not None:
            esg_7["esg_DPU_yearly_in"] = r.get("esg_DPU7_yearly_in")

        esg_by_window.setdefault(win, {})[key_tuple] = {
            "esg_5": esg_5,
            "esg_7": esg_7,
        }

        # --- VaR/CVaR settimanale (in-sample/next2) per 5 e 7 anni
        risk_5 = {}
        if r.get("risk_MEU5_in") is not None:
            risk_5["risk_MEU_in"] = r.get("risk_MEU5_in")
        if r.get("risk_DPU5_in") is not None:
            risk_5["risk_DPU_in"] = r.get("risk_DPU5_in")
        if r.get("risk_MEU5_next2") is not None:
            risk_5["risk_MEU_next2"] = r.get("risk_MEU5_next2")
        if r.get("risk_DPU5_next2") is not None:
            risk_5["risk_DPU_next2"] = r.get("risk_DPU5_next2")

        risk_7 = {}
        if r.get("risk_MEU7_in") is not None:
            risk_7["risk_MEU_in"] = r.get("risk_MEU7_in")
        if r.get("risk_DPU7_in") is not None:
            risk_7["risk_DPU_in"] = r.get("risk_DPU7_in")

        if risk_5 or risk_7:
            risk_by_window.setdefault(win, {})[key_tuple] = {
                "risk_5": risk_5,
                "risk_7": risk_7,
            }

    # mappa posizioni colonna per ciascuna finestra (0..n-1)
    assets_by_window = {}
    for w in window_stats:
        assets_by_window[w["win_tag"]] = list(range(w["n_assets"]))

    payload = {
        "base_seed": int(BASE_SEED),
        "weights": weights,
        "perf_by_window": perf_by_window,
        "scores_by_window": scores_by_window,
        "esg_by_window": esg_by_window,
        "risk_by_window": risk_by_window,
        "assets_by_window": assets_by_window,
        "param_grids": {
            "m_list": m_list,
            "lmbd_fin_list": lmbd_fin_list,
            "lesg_factors": lesg_factors,
        },
        "errors": err,
    }

    with open("cv_weights_5plus2.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Salvato: cv_weights_5plus2.pkl")
