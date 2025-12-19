import pickle 
import pandas as pd
import numpy as np

def compute_all_weight_differences(weights_dict):
    """
    Costruisce una tabella con i valori per TUTTE le finestre ws5plus2.
    
    Output DataFrame con colonne:
        - window_key  (es. 'ws5plus2_win_2019_2023_to_2025')
        - m
        - lambda_esg
        - sum_abs_diff_MEU
        - sum_abs_diff_DPU
    """

    rows = []

    for window_key, window_data in weights_dict.items():

        # Considera solo le chiavi che iniziano con 'ws5plus2'
        if not window_key.startswith("ws5plus2"):
            continue

        for params in window_data.values():

            m = params['m']
            lmbd_esg = params['lmbd_esg']

            w_meu_5 = np.array(params['w_MEU_5'])
            w_meu_7 = np.array(params['w_MEU_7'])
            w_dpu_5 = np.array(params['w_DPU_5'])
            w_dpu_7 = np.array(params['w_DPU_7'])

            diff_meu = np.sum(np.abs(w_meu_7 - w_meu_5))
            diff_dpu = np.sum(np.abs(w_dpu_7 - w_dpu_5))

            rows.append([window_key, m, lmbd_esg, diff_meu, diff_dpu])

    return pd.DataFrame(
        rows,
        columns=['window_key', 'm', 'lambda_esg', 'sum_abs_diff_MEU', 'sum_abs_diff_DPU']
    )
def aggregate_differences(df):
    """
    Ritorna un DataFrame con media e varianza dei valori
    raggruppando per (m, lambda_esg).
    """

    grouped = (
        df.groupby(['m', 'lambda_esg'])
          .agg(
              mean_MEU=('sum_abs_diff_MEU', 'mean'),
              var_MEU=('sum_abs_diff_MEU', 'var'),
              mean_DPU=('sum_abs_diff_DPU', 'mean'),
              var_DPU=('sum_abs_diff_DPU', 'var')
          )
          .reset_index()
    )

    return grouped
with open('cv_weights_5plus2.pkl','rb') as e:
    data_meanVar=pickle.load(e)
df_all_meanVar = compute_all_weight_differences(data_meanVar['weights'])
df_agg_meanVar= aggregate_differences(df_all_meanVar)
VAR=df_agg_meanVar.to_latex(float_format="%.4f",index=False )
with open('cv_weights_5plus2_CVaR.pkl','rb') as e:
    data_meanCVar=pickle.load(e)
df_all_meanCVar = compute_all_weight_differences(data_meanCVar['weights'])
df_agg_meanCVar= aggregate_differences(df_all_meanCVar)
CVAR=df_agg_meanCVar.to_latex(float_format="%.4f",index=False )