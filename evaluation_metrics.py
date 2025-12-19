import pickle
import pandas as pd 
with open('cv_weights_nested.pkl','rb') as e:
    data_1=pickle.load(e)
data_2=pd.read_csv('/Users/giorgio/Downloads/parallel_meu_dpu_results_VaR.csv')       
import math
import pandas as pd
from collections import defaultdict

# --- utilità per normalizzare i valori (gestisce Series/NaN/None/str numeriche) ---
def clean_value(raw):
    if isinstance(raw, pd.Series):
        s = pd.to_numeric(raw, errors='coerce').dropna()
        return float(s.mean()) if not s.empty else None
    try:
        v = float(raw)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None

def means_by_innermost(data, windows=(5,6,8,9,10),
                       field_meu="esg_MEU_yearly_next2",
                       field_dpu="esg_DPU_yearly_next2"):
    rows = []

    for w in windows:
        sums = defaultdict(lambda: {"MEU_sum": 0.0, "MEU_n": 0, "DPU_sum": 0.0, "DPU_n": 0})
        wdata = data.get("esg_by_window", {}).get(w, {})

        # Struttura attesa: data['esg_by_window'][w][key][inner_key] = esg_data(dict/Series)
        for sub in wdata.values():                 # livello 'key'
            for inner_key, esg_data in sub.items():  # livello 'inner_key' (più interno)
                # MEU
                v_meu = clean_value(esg_data.get(field_meu))
                if v_meu is not None:
                    sums[inner_key]["MEU_sum"] += v_meu
                    sums[inner_key]["MEU_n"] += 1
                # DPU
                v_dpu = clean_value(esg_data.get(field_dpu))
                if v_dpu is not None:
                    sums[inner_key]["DPU_sum"] += v_dpu
                    sums[inner_key]["DPU_n"] += 1

        # costruisci righe per questo window
        for ik, agg in sums.items():
            rows.append({
                "window": w,
                "inner_key": ik,
                "MEU": (agg["MEU_sum"] / agg["MEU_n"]) if agg["MEU_n"] > 0 else float("nan"),
                "DPU": (agg["DPU_sum"] / agg["DPU_n"]) if agg["DPU_n"] > 0 else float("nan"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["window", "inner_key"]).set_index(["window", "inner_key"])
    return df

# --- usa la funzione ---
df_inner = means_by_innermost(data_1, windows=(5,6,8,9,10))
print(df_inner)


import numpy as np
import pandas as pd
import math
from typing import Iterable, Dict, Any
# --- la tua funzione invariata (serve numpy) ---
# (se non l'hai già definita nello scope, incollala qui)
def portfolio_stats(
    returns: Iterable[float],
    rf: float = 0.0,          # tasso risk-free per periodo (per Sharpe)
    target: float = 0.0,      # soglia per Sortino e Omega (spesso 0 o rf)
    ann_factor: float | None = None,  # p.es. 252 per giornalieri, 12 per mensili
    ddof: int = 1             # ddof=1 -> varianza campionaria
) -> Dict[str, Any]:
    """
    Calcola metriche su una serie di ritorni discreti per periodo.

    Parametri
    ---------
    returns : iterabile di float
        Ritorni per periodo (es. [0.01, -0.02, 0.005, ...]).
    rf : float
        Tasso risk-free per periodo usato nello Sharpe.
    target : float
        Soglia di “non perdita” per Sortino e Omega (0 per rendimento assoluto,
        oppure rf se vuoi misurare rispetto al risk-free).
    ann_factor : float | None
        Fattore di annualizzazione (es. 252 per daily, 12 per monthly). Se fornito,
        aggiunge metriche annualizzate.
    ddof : int
        Gradi di libertà per la varianza (1 = campionaria, 0 = popolazione).

    Ritorna
    -------
    dict con chiavi:
      - mean, variance, min, max, cumulative_return
      - sharpe, sortino, omega
      - (opzionale) mean_ann, variance_ann, std_ann, sharpe_ann, sortino_ann
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]  # rimuove eventuali NaN
    if r.size == 0:
        return {k: np.nan for k in [
            "mean","variance","min","max","cumulative_return","sharpe","sortino","omega"
        ]}

    # Statistiche di base (per periodo)
    mean = float(np.mean(r))
    variance = float(np.var(r, ddof=ddof))
    std = float(np.sqrt(variance))
    r_min = float(np.min(r))
    r_max = float(np.max(r))
    cumulative_return = float(np.prod(1.0 + r) - 1.0)

    # Sharpe (per periodo)
    excess_mean = mean - rf
    sharpe = excess_mean / std if std > 0 else np.nan

    # Sortino (per periodo): deviazione standard downside rispetto a 'target'
    downside = np.minimum(0.0, r - target)
    # semivarianza “popolazione” (media dei quadrati degli shortfall)
    semi_variance = float(np.mean(downside ** 2))
    downside_std = float(np.sqrt(semi_variance))
    sortino = (mean - target) / downside_std if downside_std > 0 else np.nan

    # Omega ratio (Keating & Shadwick): E[(r - target)^+] / E[(target - r)^+]
    pos_part = np.maximum(r - target, 0.0)
    neg_part = np.maximum(target - r, 0.0)
    num = float(np.mean(pos_part))
    den = float(np.mean(neg_part))
    if den == 0.0 and num == 0.0:
        omega = np.nan
    elif den == 0.0:
        omega = np.inf
    else:
        omega = num / den

    out = {
        "mean": mean,
        "variance": variance,
        "min": r_min,
        "max": r_max,
        "cumulative_return": cumulative_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "omega": omega,
    }

    # Annualizzazione (se richiesta)
    if ann_factor is not None and ann_factor > 0:
        mean_ann = mean * ann_factor
        variance_ann = variance * ann_factor
        std_ann = std * np.sqrt(ann_factor)

        sharpe_ann = sharpe * np.sqrt(ann_factor) if np.isfinite(sharpe) else np.nan
        sortino_ann = sortino * np.sqrt(ann_factor) if np.isfinite(sortino) else np.nan

        out.update({
            "mean_ann": float(mean_ann),
            "variance_ann": float(variance_ann),
            "std_ann": float(std_ann),
            "sharpe_ann": float(sharpe_ann),
            "sortino_ann": float(sortino_ann),
        })

    return out


def _to_1d_float_array(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        arr = pd.to_numeric(x, errors='coerce').to_numpy()
    elif isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
    else:
        # scalare o None
        try:
            return np.array([float(x)], dtype=float)
        except (TypeError, ValueError):
            return np.array([], dtype=float)
    return arr[~np.isnan(arr)]

def compute_perf_table_annualized(
    data: dict,
    windows=(5,6,8,9,10),
    fields=("perf_MEU_next2", "perf_DPU_next2"),   # <-- non cumulati
    rf: float = 0.0,
    target: float = 0.0,
    ann_factor: float = 12,                        # es. 252 daily, 12 monthly
    ddof: int = 1,
):
    """
    Ritorna un DataFrame con indice (window, inner_key) e colonne multi-index (field, metric),
    contenente SOLO metriche annualizzate calcolate su serie NON cumulative.
    """
    if ann_factor is None or ann_factor <= 0:
        raise ValueError("Per metriche annualizzate è necessario specificare un ann_factor > 0.")

    wanted_metrics = ("mean_ann","variance_ann","std_ann","sharpe_ann","sortino_ann")

    rows = []
    index_tuples = []

    for w in windows:
        wdata = data.get("perf_by_window", {}).get(w, {})
        # atteso: data['perf_by_window'][w][key][inner_key][field] -> Series/array di rendimenti per-periodo
        for sub in wdata.values():                       # livello 'key'
            for inner_key, perf_dict in sub.items():     # livello 'inner_key'
                row = {}
                for field in fields:
                    raw = perf_dict.get(field, None)
                    arr = _to_1d_float_array(raw)        # già per-periodo (NON cumulati)
                    if arr.size == 0:
                        stats = {m: np.nan for m in wanted_metrics}
                    else:
                        stats_full = portfolio_stats(
                            returns=arr,
                            rf=rf,
                            target=target,
                            ann_factor=ann_factor,       # <-- calcola ann metriche
                            ddof=ddof
                        )
                        stats = {m: stats_full.get(m, np.nan) for m in wanted_metrics}

                    # scrivi colonne (field, metric)
                    for m in wanted_metrics:
                        row[(field, m)] = stats[m]

                if row:
                    rows.append(row)
                    index_tuples.append((w, inner_key))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index_tuples, names=["window","inner_key"]))
    df = df.reindex(sorted(df.columns, key=lambda x: (x[0], x[1])), axis=1)
    return df

# ---- Esempio d'uso ----
# Se i tuoi dati sono mensili:
df_perf_ann = compute_perf_table_annualized(
    data_1,
    windows=(5,6,8,10),
    fields=("perf_MEU_next2", "perf_DPU_next2"),  # non cumulati
    rf=0.0,
    target=0.0,
    ann_factor=12,                                 # 12 = mensile; usa 252 per giornaliero
    ddof=1
)

print(df_perf_ann.head())
df_mean = df_perf_ann.groupby(['window', 'inner_key']).mean().reset_index()

print(df_mean)
df_mean['key1'] = df_mean['inner_key'].apply(lambda x: x[0])

df_mean = df_mean.assign(key1=df_mean['inner_key'].str[0])

# medie per key1 su sole colonne numeriche
out = (
    df_mean
    .assign(key1=df_mean['inner_key'].str[0])
    .groupby(['window', 'key1'], as_index=False)
    .mean(numeric_only=True)
)
print(out)
# Ordine desiderato: mean → std → sharpe → sortino
metric_order = ['mean_ann', 'std_ann', 'sharpe_ann', 'sortino_ann']
metric_rank = {m: i for i, m in enumerate(metric_order)}

fixed_cols = ['window', 'key1']
multi_cols = [col for col in out.columns if isinstance(col, tuple)]

def sort_key(col):
    first, second = col
    # metrica non trovata → mandala in fondo
    return (metric_rank.get(second, len(metric_rank)), first)

sorted_multi_cols = sorted(multi_cols, key=sort_key)
new_col_order = fixed_cols + sorted_multi_cols

out = out[new_col_order]

print(out.columns.to_list())

a=out.iloc[:,:-2].to_latex(float_format="%.2f",index=False)


####Concentration Plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
y_meu_col = "hhi_meu"
y_dpu_col = "hhi_dpu"
expected_ratios = {0.5: "0.5×", 1.0: "1×", 2.0: "2×"}

# --- Palette colori ---
colors_meu = ["#6baed6", "#2171b5", "#08306b"]  # blu chiaro → scuro
colors_dpu = ["#fcae91", "#fb6a4a", "#cb181d"]  # rosso chiaro → scuro

# --- Prepara il dataframe ---
data = data_2.copy()
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["lmbd_fin", "lmbd_esg"])
data = data[data["lmbd_fin"] != 0]
data["ratio"] = (data["lmbd_esg"] / data["lmbd_fin"]).round(2)

# --- Imposta figure e assi (uno per ogni m) ---
unique_m = sorted(data["m"].unique())
n_m = len(unique_m)
ncols = 3  # ad esempio 3 colonne e 2 righe se hai 6 m
nrows = int(np.ceil(n_m / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=True, sharey=True)
axes = axes.flatten()  # per iterare facilmente

# --- Plot per ogni m ---
for i, (m_val, ax) in enumerate(zip(unique_m, axes)):
    df_m = data[data["m"] == m_val]

    for (r, label), c_meu, c_dpu in zip(expected_ratios.items(), colors_meu, colors_dpu):
        g = df_m[df_m["ratio"].eq(r)].sort_values("lmbd_fin")
        if g.empty:
            continue

        ax.plot(g["lmbd_fin"], g[y_meu_col], marker="o", linestyle="-", color=c_meu, label=f"MEU {label}")
        ax.plot(g["lmbd_fin"], g[y_dpu_col], marker="x", linestyle="--", color=c_dpu, label=f"DPU {label}")

    ax.set_title(f"m = {m_val}")
    ax.grid(True, alpha=0.3)

# --- Etichette e legenda generale ---
for ax in axes[n_m:]:
    ax.axis("off")  # spegni eventuali subplot vuoti

fig.text(0.5, 0.04, "λ_fin", ha="center", fontsize=12)
fig.text(0.04, 0.5, "Indice HHI", va="center", rotation="vertical", fontsize=12)

# legenda globale (fuori dai singoli plot)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


