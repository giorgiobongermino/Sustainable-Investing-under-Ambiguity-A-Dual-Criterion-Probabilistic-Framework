
# launcher_parallel_meu_dpu.py
# launcher_parallel_meu_dpu.py
import os, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

from dpu_worker import run_one_dpu_job  # worker definito nel file separato

# ========== 1) CARICAMENTO & PREP DATI ==========
data_ESG = pd.read_excel('sp100.xlsx', sheet_name='ESG SCORE',   skiprows=4)
data_fin = pd.read_excel('sp100.xlsx', sheet_name='WEEKLY PRICES', skiprows=4)

# Prima colonna come indice
data_ESG = data_ESG.set_index(data_ESG.columns[0])
data_fin = data_fin.set_index(data_fin.columns[0])

# Tieni solo le colonne ESG con >= 17 osservazioni e allinea data_fin
data_ESG = data_ESG.iloc[3:]
mask_data = (data_ESG.count() >= 16)
pos_hold  = np.where(mask_data)[0]
data_ESG  = data_ESG.loc[:, mask_data]
data_ESG=data_ESG.fillna(method='ffill')
data_fin  = data_fin.iloc[:, pos_hold]
# ESG: rimuovi le prime 3 righe e trasforma l'indice (anni -> datetime)

fin_to_drop=np.unique(np.where(data_fin.isna())[1])
data_fin = data_fin.drop(columns=data_fin.columns[fin_to_drop])
returns=data_fin.pct_change().dropna()
returns=returns*100
data_ESG = data_ESG.drop(columns=data_ESG.columns[fin_to_drop])

data_ESG.index = pd.to_datetime(data_ESG.index.astype(str), format='%Y')
#data_ESG=data_ESG/100
# Statistiche globali (μ e Σ)
mean_ESG = data_ESG.mean()
cov_ESG  = data_ESG.cov()
mean_ret = returns.mean()
cov_ret  = returns.cov()
# Array NumPy puri
mu_fin_np    = mean_ret.values.astype(float)
mu_esg_np    = mean_ESG.values.astype(float)
Sigma_fin_np = cov_ret.values.astype(float)
Sigma_esg_np = cov_ESG.values.astype(float)

n_assets = mu_fin_np.shape[0]

# ====== scale k (come tuo codice) e ν coerenti (SNR-based) ======
k = float(np.mean(mu_esg_np / np.sqrt(np.diag(Sigma_esg_np))))

diag_fin = np.clip(np.diag(Sigma_fin_np), 1e-12, np.inf)
diag_esg = np.clip(np.diag(Sigma_esg_np), 1e-12, np.inf)
#snr_fin  = float(np.mean(1/ np.sqrt(diag_fin)) )
#snr_esg  = float(np.mean(1 / np.sqrt(diag_esg)))
snr_fin  = float(np.mean(np.abs(mu_fin_np) / np.sqrt(diag_fin)))
snr_esg  = float(np.mean(np.abs(mu_esg_np) / np.sqrt(diag_esg)))


#nu_esg = float(data_ESG.shape[0]* snr_esg)
#nu_fin = float(data_fin.shape[0] * snr_fin)
nu_esg = float(np.sqrt(data_ESG.shape[0]* snr_esg))
nu_fin = float(np.sqrt(data_fin.shape[0] * snr_fin))

print(f"[INFO] n_assets={n_assets} | k={k:.2f} | snr_fin={snr_fin:.8g} | snr_esg={snr_esg:.2g}|nu_fin={nu_fin:.2g} | snr_esg={nu_esg:.2g}")

# ========== 2) GRIGLIA PARAMETRI ==========
m_list        = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
lmbd_fin_list = [0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9, 1.0,]

def lesg_triplet(lfin: float):
    """tre casi: tre quarti, uguale, minimo tra 1 e 1.5 volte (dedup robusto)"""
    vals = [0.75*lfin, lfin, min(1.5*lfin,1)]
    out = []
    for v in vals:
        if not any(np.isclose(v, u, rtol=1e-12, atol=1e-12) for u in out):
            out.append(v)
    return out

combos = []
for i_m, m in enumerate(m_list):
    for i_f, lfin in enumerate(lmbd_fin_list):
        for i_e, lesg in enumerate(lesg_triplet(lfin)):
            key = f"m_{i_m}_lmbdfin_{i_f}_lmbdesg_{i_e}"
            combos.append((key, m, lfin, lesg))

print(f"[INFO] combinazioni totali: {len(combos)}")

# ========== 3) COSTRUZIONE JOBS (solo numeri/array) ==========
# opzionale: seed per riproducibilità (commenta se non serve)
rng = np.random.default_rng(0)
jobs = []
for key, m, lfin, lesg in combos:
    jobs.append({
        "key": key,
        "m": float(m), "lmbd_fin": float(lfin), "lmbd_esg": float(lesg),
        "k": float(k), "n_assets": int(n_assets),
        "mu_fin_np": mu_fin_np, "mu_esg_np": mu_esg_np,
        "Sigma_fin_np": Sigma_fin_np, "Sigma_esg_np": Sigma_esg_np,
        "nu_fin": float(nu_fin), "nu_esg": float(nu_esg),
        "rng_seed": int(rng.integers(0, 2**31 - 1)),  # usalo se expectation_DPU lo supporta
    })

# ========== 4) ESECUZIONE PARALLELA ==========
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ctx = multiprocessing.get_context("spawn")

    N_CORES = os.cpu_count() or 8
    # con XLA single-thread per processo nel worker → puoi arrivare verso N_CORES
    N_PROCESSES = min(N_CORES, len(jobs))
    print(f"[INFO] lancio con max_workers={N_PROCESSES} su {N_CORES} core")

    results = []
    with ProcessPoolExecutor(max_workers=N_PROCESSES, mp_context=ctx) as ex:
        futs = [ex.submit(run_one_dpu_job, cfg) for cfg in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())

    ok_rows  = [r for r in results if r.get("ok")]
    err_rows = [r for r in results if not r.get("ok")]

    # ========== 5) SALVATAGGIO METRICHE ==========
    df = pd.DataFrame(ok_rows).sort_values(["m", "lmbd_fin", "lmbd_esg"])
    df.to_csv("parallel_meu_dpu_results.csv", index=False)
    df.to_parquet("parallel_meu_dpu_results.parquet", index=False)
    if err_rows:
        pd.DataFrame(err_rows).to_csv("parallel_meu_dpu_errors.csv", index=False)

    print("Salvati:")
    print(" - parallel_meu_dpu_results.[csv|parquet]")
    if err_rows:
        print(f" - parallel_meu_dpu_errors.csv (errori: {len(err_rows)})")

    # ========== 6) SALVATAGGIO DIZIONARI PESI ==========
    weights_meu = {r["key"]: np.asarray(r["w_MEU"]) for r in ok_rows}
    weights_dpu = {r["key"]: np.asarray(r["w_DPU"]) for r in ok_rows}

    np.savez_compressed("weights_meu.npz", **weights_meu)
    np.savez_compressed("weights_dpu.npz", **weights_dpu)

    # opzionale: anche pickle
    import pickle
    with open("weights_meu.pkl", "wb") as f:
        pickle.dump(weights_meu, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("weights_dpu.pkl", "wb") as f:
        pickle.dump(weights_dpu, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(" - weights_meu.npz | weights_dpu.npz  (+ .pkl opzionali)")
    print(df.head())
