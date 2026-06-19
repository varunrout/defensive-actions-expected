from __future__ import annotations
import numpy as np
import pandas as pd

def first_existing(df, names):
    return next((n for n in names if n in df.columns), None)


def add_suppression(df: pd.DataFrame) -> pd.DataFrame:
    """Build canonical coach suppression fields using expected - observed sign convention."""
    out = df.copy()
    required = (
        'coach_expected_shot_probability',
        'coach_observed_future_shot',
        'coach_expected_future_xg_r4',
        'coach_expected_future_xg_two_part',
        'coach_observed_future_xg',
    )
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"Cannot compute coach suppression metrics. Missing columns: {missing}")
    out['coach_shot_suppression'] = out['coach_expected_shot_probability'] - out['coach_observed_future_shot']
    out['coach_xg_suppression_r4'] = out['coach_expected_future_xg_r4'] - out['coach_observed_future_xg']
    out['coach_xg_suppression_two_part'] = out['coach_expected_future_xg_two_part'] - out['coach_observed_future_xg']
    return out

def bootstrap_ci(values, statistic=np.mean, n_boot=500, seed=7):
    arr=np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(arr)==0: return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed); stats=[statistic(rng.choice(arr, len(arr), True)) for _ in range(n_boot)]
    return (float(statistic(arr)), float(np.quantile(stats,.025)), float(np.quantile(stats,.975)))


def bootstrap_match_level_ci(
    df: pd.DataFrame,
    value_col: str,
    *,
    match_col: str = 'match_id',
    n_boot: int = 500,
    seed: int = 7,
) -> tuple[float, float, float]:
    if value_col not in df.columns or match_col not in df.columns or df.empty:
        return (np.nan, np.nan, np.nan)
    by_match = df.groupby(match_col, dropna=False)[value_col].mean().dropna()
    return bootstrap_ci(by_match.values, n_boot=n_boot, seed=seed)

def summary_table(df, group_cols):
    if df.empty:
        return pd.DataFrame(columns=list(group_cols) + ['actions', 'matches', 'players'])
    agg = {
        'actions': ('event_id', 'size') if 'event_id' in df.columns else (df.columns[0], 'size'),
        'matches': ('match_id', 'nunique') if 'match_id' in df.columns else (df.columns[0], 'size'),
        'players': ('player', 'nunique') if 'player' in df.columns else (df.columns[0], 'size'),
    }
    optional = {
        'observed_shot_rate': 'coach_observed_future_shot',
        'expected_shot_rate': 'coach_expected_shot_probability',
        'observed_xg': 'coach_observed_future_xg',
        'expected_xg_r4': 'coach_expected_future_xg_r4',
        'expected_xg_two_part': 'coach_expected_future_xg_two_part',
        'shot_suppression': 'coach_shot_suppression',
        'xg_suppression_r4': 'coach_xg_suppression_r4',
        'xg_suppression_two_part': 'coach_xg_suppression_two_part',
        'possession_win_rate': 'action_won_possession',
        'visibility_coverage': 'coach_reliable_visibility',
    }
    for output_name, column in optional.items():
        if column in df.columns:
            agg[output_name] = (column, 'mean')
    base = df.groupby(list(group_cols), dropna=False).agg(**agg).reset_index()
    if 'coach_shot_suppression' in df.columns and 'match_id' in df.columns:
        ci_rows = []
        for _, group in df.groupby(list(group_cols), dropna=False):
            mean, lower, upper = bootstrap_match_level_ci(group, 'coach_shot_suppression')
            ci_rows.append({'shot_suppression_ci_lower': lower, 'shot_suppression_ci_upper': upper, 'shot_suppression_boot_mean': mean})
        base = pd.concat([base.reset_index(drop=True), pd.DataFrame(ci_rows)], axis=1)
    base['minimum_sample_flag'] = base['actions'] < 30
    return base.sort_values('actions', ascending=False)
