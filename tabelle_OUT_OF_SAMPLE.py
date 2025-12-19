import numpy as np
import pandas as pd
import pickle
from typing import Iterable, Dict, Any
import math
from collections import defaultdict

# ---------- utilities ----------
def _to_1d_float_array(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        arr = pd.to_numeric(x, errors='coerce').to_numpy()
    elif isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
    else:
        try:
            return np.array([float(x)], dtype=float)
        except (TypeError, ValueError):
            return np.array([], dtype=float)
    return arr[~np.isnan(arr)]

def portfolio_stats(
    returns: Iterable[float],
    rf: float = 0.0,
    target: float = 0.0,
    ann_factor: float | None = None,
    ddof: int = 1
) -> Dict[str, Any]:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return {k: np.nan for k in [
            "mean","variance","min","max","cumulative_return","sharpe","sortino","omega",
            "mean_ann","variance_ann","std_ann","sharpe_ann","sortino_ann"
        ]}
    mean = float(np.mean(r))
    variance = float(np.var(r, ddof=ddof))
    std = float(np.sqrt(variance))
    excess_mean = mean - rf
    sharpe = excess_mean / std if std > 0 else np.nan
    downside = np.minimum(0.0, r - target)
    semi_variance = float(np.mean(downside ** 2))
    downside_std = float(np.sqrt(semi_variance))
    sortino = (mean - target) / downside_std if downside_std > 0 else np.nan

    out = {
        "mean": mean, "variance": variance, "min": float(np.min(r)),
        "max": float(np.max(r)), "cumulative_return": float(np.prod(1+r) - 1),
        "sharpe": sharpe, "sortino": sortino,
    }
    if ann_factor and ann_factor > 0:
        out.update({
            "mean_ann": float(mean * ann_factor),
            "variance_ann": float(variance * ann_factor),
            "std_ann": float(std * np.sqrt(ann_factor)),
            "sharpe_ann": float(sharpe * np.sqrt(ann_factor)) if np.isfinite(sharpe) else np.nan,
            "sortino_ann": float(sortino * np.sqrt(ann_factor)) if np.isfinite(sortino) else np.nan,
        })
    return out

# ---------- core builders ----------
def _from_nested_dict_to_perf_table(
    data: dict,
    windows=(5,6,8,9,10),
    fields=("perf_MEU_next2","perf_DPU_next2"),
    rf=0.0, target=0.0, ann_factor=12, ddof=1
) -> pd.DataFrame:
    """Costruisce la tabella (window, inner_key) × (field, metric) dal dict annidato."""
    wanted = ("mean_ann","variance_ann","std_ann","sharpe_ann","sortino_ann")
    rows, idx = [], []
    for w in windows:
        wdata = data.get("perf_by_window", {}).get(w, {})
        for sub in wdata.values():                          # key
            for inner_key, perf_dict in sub.items():        # inner_key
                row = {}
                for f in fields:
                    arr = _to_1d_float_array(perf_dict.get(f))
                    stats = portfolio_stats(arr, rf, target, ann_factor, ddof) if arr.size>0 \
                            else {m: np.nan for m in wanted}
                    for m in wanted:
                        row[(f,m)] = stats[m]
                rows.append(row); idx.append((w, inner_key))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=["window","inner_key"]))
    df = df.reindex(sorted(df.columns, key=lambda x: (x[0], x[1])), axis=1)
    return df.reset_index()

def _from_dataframe_to_perf_table(
    df: pd.DataFrame,
    windows=(5,6,8,9,10),
    fields=("perf_MEU_next2","perf_DPU_next2"),
    rf=0.0, target=0.0, ann_factor=12, ddof=1
) -> pd.DataFrame:
    """
    Variante se i dati sono già in un DataFrame con colonne:
    - 'window', 'inner_key', e per ciascun field una lista/Series di rendimenti per-periodo.
    """
    wanted = ("mean_ann","variance_ann","std_ann","sharpe_ann","sortino_ann")
    take_cols = ['window','inner_key'] + list(fields)
    if not set(take_cols).issubset(df.columns):
        missing = set(take_cols) - set(df.columns)
        raise ValueError(f"Mancano nel DataFrame le colonne: {missing}")

    rows = []
    for _, row in df[df['window'].isin(windows)][take_cols].iterrows():
        out_row = {}
        for f in fields:
            arr = _to_1d_float_array(row[f])
            stats = portfolio_stats(arr, rf, target, ann_factor, ddof) if arr.size>0 \
                    else {m: np.nan for m in wanted}
            for m in wanted: out_row[(f,m)] = stats[m]
        rows.append({'window':row['window'], 'inner_key':row['inner_key'], **out_row})
    out = pd.DataFrame(rows)
    # ordina multi-colonne
    multi_cols = [c for c in out.columns if isinstance(c, tuple)]
    out = out[['window','inner_key'] + sorted(multi_cols, key=lambda x:(x[0],x[1]))]
    return out

def _aggregate_and_flatten(out: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Media per (window, key1) + ordinamento metriche + header piatto + LaTeX."""
    # key1 = prima lettera di inner_key
    out['key1'] = out['inner_key'].astype(str).str[0]
    agg = out.groupby(['window','key1'], as_index=False).mean(numeric_only=True)

    metric_order = ['mean_ann','std_ann','sharpe_ann','sortino_ann']
    metric_rank = {m:i for i,m in enumerate(metric_order)}
    multi_cols = [c for c in agg.columns if isinstance(c, tuple)]
    multi_cols_sorted = sorted(multi_cols, key=lambda c:(metric_rank.get(c[1],99), c[0]))

    agg = agg[['window','key1'] + multi_cols_sorted]

    def nice_field(f):  # 'perf_MEU_next2' -> 'MEU'
        return 'MEU' if 'MEU' in f else 'DPU'
    def nice_metric(m):
        mm = m.replace('_ann','')
        return {'mean':'Mean','std':'Std','sharpe':'Sharpe','sortino':'Sortino'}.get(mm, mm.title())

    flat_cols = ['window','key1'] + [f"{nice_field(f)} {nice_metric(m)}" for f,m in multi_cols_sorted]
    agg_flat = agg.copy()
    agg_flat.columns = flat_cols

    latex = agg_flat.to_latex(index=False, float_format="%.2f", escape=False)
    return latex, agg_flat

# --- helper per flatten senza aggregazione ---
def _aggregate_by_inner_first_and_flatten(base: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Aggrega facendo la media per (window, first(inner_key)),
    dove first(inner_key) è il primo elemento della tupla inner_key.
    Appiattisce le colonne (field, metric) in etichette pulite.
    """
    if not {'window','inner_key'}.issubset(base.columns):
        base = base.reset_index()

    # estrae il primo elemento della tupla inner_key
    def first_of_tuple(x):
        if isinstance(x, tuple) and len(x) > 0:
            return x[0]
        # se non è tupla, lo usa così com'è
        return x

    base = base.copy()
    base['inner_first'] = base['inner_key'].apply(first_of_tuple)

    # media numerica per (window, inner_first)
    agg = base.groupby(['window', 'inner_first'], as_index=False).mean(numeric_only=True)

    # ordina metriche come desideri
    metric_order = ['mean_ann', 'std_ann', 'sharpe_ann', 'sortino_ann']
    metric_rank = {m: i for i, m in enumerate(metric_order)}
    multi_cols = [c for c in agg.columns if isinstance(c, tuple)]
    multi_cols_sorted = sorted(multi_cols, key=lambda c: (metric_rank.get(c[1], 99), c[0]))

    # seleziona colonne finali
    agg = agg[['window', 'inner_first'] + multi_cols_sorted]

    # nomi colonna leggibili
    def nice_field(f):   # 'perf_MEU_next2' -> 'MEU' / 'DPU'
        return 'MEU' if 'MEU' in f else 'DPU'
    def nice_metric(m):
        mm = m.replace('_ann','')
        return {'mean': 'Mean', 'std': 'Std', 'sharpe': 'Sharpe', 'sortino': 'Sortino'}.get(mm, mm.title())

    flat_cols = ['window', 'inner_first'] + [f"{nice_field(f)} {nice_metric(m)}" for f, m in multi_cols_sorted]
    agg_flat = agg.copy()
    agg_flat.columns = flat_cols

    latex = agg_flat.to_latex(index=False, float_format="%.2f", escape=False)
    return latex, agg_flat


def make_perf_table(
    raw,                                  # dict annidato OPPURE DataFrame
    windows=(5,6,8,10),
    fields=("perf_MEU_next2","perf_DPU_next2"),
    rf=0.0, target=0.0, ann_factor=12, ddof=1,
    aggregate_by_key1: bool = False,         # vecchio flag (prima lettera di inner_key come stringa)
    aggregate_by_inner_first: bool = False   # NUOVO: primo elemento della tupla inner_key
):
    """
    Ritorna: (latex_string, table_dataframe)

    Modalità:
      - aggregate_by_inner_first=True  -> media per (window, first(inner_key_tuple))
      - aggregate_by_key1=True         -> media per (window, prima lettera di inner_key come stringa)
      - entrambi False (default)       -> nessuna media: una riga per (window, inner_key)
    """
    if isinstance(raw, dict):
        base = _from_nested_dict_to_perf_table(raw, windows, fields, rf, target, ann_factor, ddof)
    elif isinstance(raw, pd.DataFrame):
        base = _from_dataframe_to_perf_table(raw, windows, fields, rf, target, ann_factor, ddof)
    else:
        raise TypeError("Parametro 'raw' deve essere un dict annidato o un pandas.DataFrame.")

    if aggregate_by_inner_first:
        latex, table = _aggregate_by_inner_first_and_flatten(base)
    elif aggregate_by_key1:
        latex, table = _aggregate_and_flatten(base)    # media per prima lettera (versione precedente)
              # nessuna media

    return latex, table

with open('cv_weights_nested_CVaR.pkl','rb') as e:
    data_meanCVar=pickle.load(e)

latexCVar, tableCVar = make_perf_table(
    raw=data_meanCVar,              # il dict del pickle (o un DataFrame)
    windows=(5,6,8,10),
    fields=("perf_MEU_next2","perf_DPU_next2"),   # rendimenti per-periodo (NON cumulati)
    ann_factor=12,                  # 12 mensile, 252 giornaliero
    aggregate_by_inner_first=True 
)
print(latexCVar)       # stringa LaTeX
tableCVar.head()       # pandas DataFrame finale (colonne piatte)
tableCVar=tableCVar.iloc[:,:-2]
latexCVar=tableCVar.to_latex(index=False,float_format="%.2f")
with open('cv_weights_nested.pkl','rb') as e:
    data_meanVar=pickle.load(e)

latexVar, tableVar = make_perf_table(
    raw=data_meanVar,              # il dict del pickle (o un DataFrame)
    windows=(5,6,8,10),
    fields=("perf_MEU_next2","perf_DPU_next2"),   # rendimenti per-periodo (NON cumulati)
    ann_factor=52 ,                 # 12 mensile, 252 giornaliero
    aggregate_by_inner_first=True 
)
print(latexVar)       # stringa LaTeX
tableVar.head()

tableVar=tableVar.iloc[:,:-2]
latexVar=tableVar.to_latex(index=False,float_format="%.2f")
