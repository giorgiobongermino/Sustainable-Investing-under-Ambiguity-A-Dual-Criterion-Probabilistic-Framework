# Sustainable-Investing-under-Ambiguity-A-Dual-Criterion-Probabilistic-Framework
Supplementary materials for the project: Sustainable Investing under Ambiguity: A Dual-Criterion Probabilistic Framework (Bongermino, Romagnoli, Rossi)
Helper scripts to compare two portfolio-construction rules:
- **MEU** (mean–variance expected utility) solved with SLSQP on the simplex.
- **DPU** (dual preference utility) estimated with NumPyro/JAX on the simplex.

All launchers read ESG scores and prices from `sp100.xlsx` (other index Excel files are available) and save grids of weights/metrics plus summary tables for out-of-sample evaluation.

## Key files
- `codice_base.py` — core primitives: simplex optimizer, NUTS-based mean on the simplex (`mean_simplex_from_logp`), DPU combiner (`expectation_DPU`), and mean–variance/ESG objectives.
- `dpu_worker*.py` — workers executed in separate processes. Base version uses mean–variance; `*_CVaR` variants use empirical mean–CVaR for the financial leg; `*_blocks` versions also build performance series, yearly ESG scores, and risk metrics; `*_5plus2` implements the 5-year in-sample + 2-year out-of-sample scheme.
- `launcher_parallel_dpu*.py` — single-window grid search over `(m, λ_1, λ_2)`; writes `parallel_meu_dpu_results*.csv|parquet` and `weights_*.[npz|pkl]`.
- `launcher_cv_parallel_multi_ws*.py` — cross-validation across multiple window sizes (e.g., 5, 6, 8, 10 years) using block workers; outputs `cv_weights_nested*.pkl`.
- `launcher_cv_parallel_5plus2*.py` — cross-val using the 5+2 design via the 5plus2 workers; outputs `cv_weights_nested_5plus2*.pkl`.
- `tabelle_OUT_OF_SAMPLE.py` / `tabella_OUT_SAMPLE_mean.py` — build LaTeX/CSV-ready performance tables from the `cv_weights_*` pickles (mean/std/Sharpe/Sortino, plus VaR/ES in the mean version).
- `evaluation_metrics.py` — utilities to aggregate ESG metrics, compute annualized performance tables, and a concentration/HHI plotting snippet.
- `plot*.py` and notebooks — exploratory plots and diagnostics (concentration, stability ratio).
