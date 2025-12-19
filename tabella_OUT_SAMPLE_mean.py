import numpy as np
import pandas as pd
import pickle
from typing import Iterable, Dict, Any

# ---------- utilities ----------
def _to_1d_float_array(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        arr = pd.to_numeric(x, errors="coerce").to_numpy()
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
    ddof: int = 1,
    alpha: float = 0.95,       # confidence level for VaR/ES
) -> Dict[str, Any]:
    """
    Calcola statistiche di portafoglio su una serie di rendimenti per-periodo.

    - mean / mean_ann: usati per Sharpe e Sortino (convenzionali).
    - VaR / ES: one-sided left tail sul rendimento (es. 95% -> 5% quantile),
      riportati come perdite positive.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return {k: np.nan for k in [
            "mean", "variance", "min", "max", "cumulative_return",
            "sharpe", "sortino",
            "mean_ann", "variance_ann", "std_ann",
            "sharpe_ann", "sortino_ann",
            "VaR", "ES",
        ]}

    # centro / dispersione
    mean = float(np.mean(r))
    variance = float(np.var(r, ddof=ddof))
    std = float(np.sqrt(variance))

    # Sharpe / Sortino
    excess_mean = mean - rf
    sharpe = excess_mean / std if std > 0 else np.nan

    downside = np.minimum(0.0, r - target)
    semi_variance = float(np.mean(downside**2))
    downside_std = float(np.sqrt(semi_variance))
    sortino = (mean - target) / downside_std if downside_std > 0 else np.nan

    # VaR & ES (positive loss convention)
    tail_p = 1.0 - alpha  # es. 0.05 se alpha=0.95
    q = float(np.quantile(r, tail_p))  # left-tail quantile of returns
    var_val = -q
    tail = r[r <= q]
    es_val = -float(tail.mean()) if tail.size > 0 else np.nan

    out: Dict[str, Any] = {
        "mean": mean,
        "variance": variance,
        "min": float(np.min(r)),
        "max": float(np.max(r)),
        "cumulative_return": float(np.prod(1 + r) - 1),
        "sharpe": sharpe,
        "sortino": sortino,
        "VaR": var_val,
        "ES": es_val,
    }

    if ann_factor is not None and ann_factor > 0:
        out.update({
            "mean_ann": float(mean * ann_factor),
            "variance_ann": float(variance * ann_factor),
            "std_ann": float(std * np.sqrt(ann_factor)),
            "sharpe_ann": float(sharpe * np.sqrt(ann_factor)) if np.isfinite(sharpe) else np.nan,
            "sortino_ann": float(sortino * np.sqrt(ann_factor)) if np.isfinite(sortino) else np.nan,
        })
    else:
        out.update({
            "mean_ann": np.nan,
            "variance_ann": np.nan,
            "std_ann": np.nan,
            "sharpe_ann": np.nan,
            "sortino_ann": np.nan,
        })

    return out


# ---------- core builders ----------
def _from_nested_dict_to_perf_table(
    data: dict,
    windows=(5, 6, 8, 9, 10),
    fields=("perf_MEU_next2", "perf_DPU_next2"),
    rf=0.0, target=0.0, ann_factor=12, ddof=1
) -> pd.DataFrame:
    """
    Costruisce la tabella (window, inner_key) × (field, metric) dal dict annidato.
    Si aspetta che `data["perf_by_window"][w]` sia un dict -> dict -> perf_dict,
    dove perf_dict contiene per ciascun `field` una Serie/lista di rendimenti per-periodo.
    """
    # metriche che vogliamo in output per ciascun field
    wanted = (
        "mean_ann", "variance_ann", "std_ann",
        "sharpe_ann", "sortino_ann",
        "VaR", "ES",
    )

    rows, idx = [], []
    for w in windows:
        wdata = data.get("perf_by_window", {}).get(w, {})
        for sub in wdata.values():  # primo livello (key)
            for inner_key, perf_dict in sub.items():  # inner_key
                row = {}
                for f in fields:
                    arr = _to_1d_float_array(perf_dict.get(f))
                    stats = (
                        portfolio_stats(arr, rf, target, ann_factor, ddof)
                        if arr.size > 0
                        else {m: np.nan for m in wanted}
                    )
                    for m in wanted:
                        row[(f, m)] = stats[m]
                rows.append(row)
                idx.append((w, inner_key))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["window", "inner_key"])
    )

    # ordina le colonne multi-indice in modo consistente
    df = df.reindex(
        sorted(df.columns, key=lambda x: (x[0], x[1])),
        axis=1
    )
    return df.reset_index()


def _from_dataframe_to_perf_table(
    df: pd.DataFrame,
    windows=(5, 6, 8, 9, 10),
    fields=("perf_MEU_next2", "perf_DPU_next2"),
    rf=0.0, target=0.0, ann_factor=12, ddof=1
) -> pd.DataFrame:
    """
    Variante se i dati sono già in un DataFrame con colonne:
    - 'window', 'inner_key', e per ciascun field una lista/Series di rendimenti per-periodo.
    """
    wanted = (
        "mean_ann", "variance_ann", "std_ann",
        "sharpe_ann", "sortino_ann",
        "VaR", "ES",
    )

    take_cols = ["window", "inner_key"] + list(fields)
    if not set(take_cols).issubset(df.columns):
        missing = set(take_cols) - set(df.columns)
        raise ValueError(f"Mancano nel DataFrame le colonne: {missing}")

    rows = []
    for _, row in df[df["window"].isin(windows)][take_cols].iterrows():
        out_row: Dict[Any, Any] = {}
        for f in fields:
            arr = _to_1d_float_array(row[f])
            stats = (
                portfolio_stats(arr, rf, target, ann_factor, ddof)
                if arr.size > 0
                else {m: np.nan for m in wanted}
            )
            for m in wanted:
                out_row[(f, m)] = stats[m]
        rows.append({"window": row["window"], "inner_key": row["inner_key"], **out_row})

    out_df = pd.DataFrame(rows)
    # ordina le colonne multi-indice in modo consistente
    multi_cols = [c for c in out_df.columns if isinstance(c, tuple)]
    out_df = out_df[["window", "inner_key"] + sorted(multi_cols, key=lambda x: (x[0], x[1]))]
    return out_df


def _metric_order_and_rank():
    metric_order = [
        "mean_ann", "std_ann", "sharpe_ann", "sortino_ann",
        "variance_ann", "VaR", "ES",
    ]
    metric_rank = {m: i for i, m in enumerate(metric_order)}
    return metric_order, metric_rank


def _nice_field(f: str) -> str:
    # 'perf_MEU_next2' -> 'MEU' / 'DPU'
    return "MEU" if "MEU" in f else "DPU"


def _nice_metric(m: str) -> str:
    mm = m.replace("_ann", "")
    return {
        "mean": "Mean",
        "std": "Std",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "variance": "Var",
        "VaR": "VaR",
        "ES": "ES",
    }.get(mm, mm.title())


def _aggregate_and_flatten(out: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Aggrega per (window, key1) dove key1 = prima lettera di inner_key.
    ATTENZIONE: l'aggregazione qui usa la MEDIANA delle metriche (VaR, Sortino, ecc.),
    non la media.
    """
    out = out.copy()
    out["key1"] = out["inner_key"].astype(str).str[0]
    # <-- QUI usiamo median invece di mean
    agg = out.groupby(["window", "key1"], as_index=False).median(numeric_only=True)

    _, metric_rank = _metric_order_and_rank()
    multi_cols = [c for c in agg.columns if isinstance(c, tuple)]
    multi_cols_sorted = sorted(multi_cols, key=lambda c: (metric_rank.get(c[1], 99), c[0]))

    agg = agg[["window", "key1"] + multi_cols_sorted]

    flat_cols = ["window", "key1"] + [
        f"{_nice_field(f)} {_nice_metric(m)}" for f, m in multi_cols_sorted
    ]
    agg_flat = agg.copy()
    agg_flat.columns = flat_cols

    latex = agg_flat.to_latex(index=False, float_format="%.2f", escape=False)
    return latex, agg_flat


def _aggregate_by_inner_first_and_flatten(base: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Aggrega facendo la mediana per (window, first(inner_key)),
    dove first(inner_key) è il primo elemento della tupla inner_key.
    Appiattisce le colonne (field, metric) in etichette pulite.
    """
    if not {"window", "inner_key"}.issubset(base.columns):
        base = base.reset_index()

    def first_of_tuple(x):
        if isinstance(x, tuple) and len(x) > 0:
            return x[0]
        return x

    base = base.copy()
    base["inner_first"] = base["inner_key"].apply(first_of_tuple)

    # <-- QUI usiamo median invece di mean
    agg = base.groupby(["window", "inner_first"], as_index=False).median(numeric_only=True)

    _, metric_rank = _metric_order_and_rank()
    multi_cols = [c for c in agg.columns if isinstance(c, tuple)]
    multi_cols_sorted = sorted(multi_cols, key=lambda c: (metric_rank.get(c[1], 99), c[0]))

    agg = agg[["window", "inner_first"] + multi_cols_sorted]

    flat_cols = ["window", "inner_first"] + [
        f"{_nice_field(f)} {_nice_metric(m)}" for f, m in multi_cols_sorted
    ]
    agg_flat = agg.copy()
    agg_flat.columns = flat_cols

    latex = agg_flat.to_latex(index=False, float_format="%.2f", escape=False)
    return latex, agg_flat


def _no_aggregate_and_flatten(base: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Nessuna aggregazione: una riga per (window, inner_key).
    Appiattisce soltanto le colonne (field, metric).
    """
    if base.empty:
        return base.to_latex(index=False), base

    _, metric_rank = _metric_order_and_rank()
    multi_cols = [c for c in base.columns if isinstance(c, tuple)]
    multi_cols_sorted = sorted(multi_cols, key=lambda c: (metric_rank.get(c[1], 99), c[0]))

    flat = base[["window", "inner_key"] + multi_cols_sorted].copy()
    flat_cols = ["window", "inner_key"] + [
        f"{_nice_field(f)} {_nice_metric(m)}" for f, m in multi_cols_sorted
    ]
    flat.columns = flat_cols

    latex = flat.to_latex(index=False, float_format="%.2f", escape=False)
    return latex, flat


def make_perf_table(
    raw,                                  # dict annidato OPPURE DataFrame
    windows=(5, 6, 8, 10),
    fields=("perf_MEU_next2", "perf_DPU_next2"),
    rf=0.0, target=0.0, ann_factor=12, ddof=1,
    aggregate_by_key1: bool = False,         # vecchio flag (prima lettera di inner_key come stringa)
    aggregate_by_inner_first: bool = False   # NUOVO: primo elemento della tupla inner_key
):
    """
    Ritorna: (latex_string, table_dataframe)

    Modalità:
      - aggregate_by_inner_first=True  -> mediana per (window, first(inner_key_tuple))
      - aggregate_by_key1=True         -> mediana per (window, prima lettera di inner_key come stringa)
      - entrambi False (default)       -> nessuna media/mediana: una riga per (window, inner_key)

    NOTA: l'aggregazione di VaR, Sortino, ecc. usa la MEDIANA delle metriche tra i vari inner_key.
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
        latex, table = _aggregate_and_flatten(base)
    else:
        latex, table = _no_aggregate_and_flatten(base)

    return latex, table


# ------------------ esempio d'uso con i pickle ------------------
if __name__ == "__main__":
    # CVaR-based weights
    with open("cv_weights_nested_CVaR.pkl", "rb") as e:
        data_meanCVar = pickle.load(e)

    latexCVar, tableCVar = make_perf_table(
        raw=data_meanCVar,
        windows=(5, 6, 8, 10),
        fields=("perf_MEU_next2", "perf_DPU_next2"),
        ann_factor=52,           # es. mensile
        aggregate_by_inner_first=True,
    )
    print("=== CVaR table ===")
    print(latexCVar)
    print(tableCVar.head())

    # Var-based weights (o media-varianza classica)
    with open("cv_weights_nested.pkl", "rb") as e:
        data_meanVar = pickle.load(e)

    latexVar, tableVar = make_perf_table(
        raw=data_meanVar,
        windows=(5, 6, 8, 10),
        fields=("perf_MEU_next2", "perf_DPU_next2"),
        ann_factor=52,           # es. settimanale
        aggregate_by_inner_first=True,
    )
    print("=== Var table ===")
    print(latexVar)
    print(tableVar.head())
# ---- helper to keep only Return, Sharpe, Sortino, ES ----
def filter_for_paper(table: pd.DataFrame) -> pd.DataFrame:
    # which metrics (by column suffix) you want in the LaTeX table
    wanted_metrics = ("Mean", "Sharpe", "Sortino")   # use "Median" instead of "Mean" if needed

    # identifier columns (whatever exists among these)
    id_cols = [c for c in ("window", "inner_first", "key1", "inner_key") if c in table.columns]

    # value columns that end with one of the wanted metric names
    value_cols = [
        c for c in table.columns
        if isinstance(c, str) and any(c.endswith(" " + w) for w in wanted_metrics)
    ]

    # final ordered table
    return table[id_cols + value_cols]


# --------- CVaR table ---------
tableCVar_paper = filter_for_paper(tableCVar)
latexCVar = tableCVar_paper.to_latex(index=False, float_format="%.2f")
print(latexCVar)

# --------- Var / mean-variance table ---------
tableVar_paper = filter_for_paper(tableVar)
latexVar = tableVar_paper.to_latex(index=False, float_format="%.2f")
print(latexVar)

def extract_var_es(table: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Estrae solo VaR ed ES e rinomina le colonne aggiungendo un suffisso (label),
    es: 'MEU VaR' -> 'MEU VaR (CVaR)'
    """
    id_cols = [c for c in ("window", "inner_first", "key1", "inner_key") if c in table.columns]

    value_cols = [
        c for c in table.columns
        if isinstance(c, str) and (c.endswith(" VaR") or c.endswith(" ES"))
    ]

    out = table[id_cols + value_cols].copy()

    rename_map = {c: f"{c} ({label})" for c in value_cols}
    out = out.rename(columns=rename_map)

    return out
var_es_CVaR = extract_var_es(tableCVar, label="CVaR")
var_es_Var  = extract_var_es(tableVar,  label="Var")

# ---- merge affiancato ----
id_cols = [c for c in ("window", "inner_first", "key1", "inner_key") if c in var_es_CVaR.columns]

var_es_side_by_side = pd.merge(
    var_es_CVaR,
    var_es_Var,
    on=id_cols,
    how="inner"
)

# ---- LaTeX ----
latex_var_es = var_es_side_by_side.to_latex(
    index=False,
    float_format="%.2f",
    escape=False
)

print("=== VaR & ES comparison table ===")
print(latex_var_es)
print(var_es_side_by_side.head())