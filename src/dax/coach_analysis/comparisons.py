from __future__ import annotations
import pandas as pd
from .metrics import summary_table

def competition_comparison(df, group_cols=None, competition_col='competition'):
    group_cols=group_cols or []
    if df.empty or competition_col not in df.columns: return pd.DataFrame()
    return summary_table(df, [competition_col, *group_cols])

def context_adjusted_difference(df, competition_col='competition', strata=None, outcome='observed_future_xg'):
    strata=strata or [c for c in ['action_family','phase','position_group'] if c in df.columns]
    if df.empty or competition_col not in df.columns or outcome not in df.columns: return pd.DataFrame()
    denom=df.groupby([competition_col,*strata], dropna=False).size().rename('actions').reset_index()
    rates=df.groupby([competition_col,*strata], dropna=False)[outcome].mean().rename('rate').reset_index()
    return denom.merge(rates, on=[competition_col,*strata])
