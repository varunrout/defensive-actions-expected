from __future__ import annotations

import pandas as pd

VISIBILITY_COLUMNS = ['has_360', 'local_5m_region_fully_visible', 'local_10m_region_fully_visible', 'freeze_frame_roles_known']


def add_reliable_visibility(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    available = [c for c in VISIBILITY_COLUMNS if c in out.columns]
    if not available:
        out['coach_reliable_visibility'] = False
        return out
    bools = out[available].fillna(False).astype(bool)
    out['coach_reliable_visibility'] = bools.all(axis=1)
    return out


def visibility_report(df: pd.DataFrame) -> dict:
    rows = int(len(df))
    report = {'rows': rows, 'columns': {}, 'reliable_rows': 0, 'reliable_rate': 0.0}
    if rows == 0:
        return report
    out = add_reliable_visibility(df)
    for col in VISIBILITY_COLUMNS:
        if col in out.columns:
            values = out[col].fillna(False).astype(bool)
            report['columns'][col] = {'true': int(values.sum()), 'false_or_missing': int((~values).sum()), 'true_rate': float(values.mean())}
        else:
            report['columns'][col] = {'missing': True}
    report['reliable_rows'] = int(out['coach_reliable_visibility'].sum())
    report['reliable_rate'] = float(out['coach_reliable_visibility'].mean())
    return report
