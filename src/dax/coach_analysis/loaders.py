from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

REQUIRED_INPUTS = {
    'actions': 'data/features/player_defensive_actions.parquet',
    'summary': 'data/features/player_defensive_summary.parquet',
    'signals': 'data/features/player_defensive_signals_provisional_oof.parquet',
    'classification_oof': 'outputs/oof/classification_oof.parquet',
    'regression_oof': 'outputs/oof/regression_oof.parquet',
    'two_part_oof': 'outputs/oof/two_part_future_xg_oof.parquet',
    'two_part_exploratory_oof': 'outputs/oof/two_part_future_xg_oof_exploratory.parquet',
    'classification_report': 'outputs/models/reports/classification_evaluation_summary.csv',
    'regression_report': 'outputs/models/reports/regression_evaluation_summary.csv',
    'two_part_report': 'outputs/models/reports/two_part_vs_one_stage_common_rows.csv',
    'sensitivity': 'outputs/models/reports/player_signal_sensitivity.csv',
    'reliability_thresholds': 'outputs/models/reports/player_signal_reliability_thresholds.json',
    'analysis_config': 'configs/analysis.yaml',
    'models_config': 'configs/models.yaml',
}
EVENT_KEYS = ['match_id', 'event_id']
@dataclass(frozen=True)
class InputStatus:
    name: str; path: str; exists: bool; rows: int | None = None; columns: int | None = None; error: str | None = None

def repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for q in [p, *p.parents]:
        if (q/'.git').exists() or (q/'pyproject.toml').exists() or (q/'requirements.txt').exists(): return q
    return p

def ensure_output_dirs(root: Path | None = None) -> dict[str, Path]:
    root = root or repo_root(); base = root/'outputs'/'coach_analysis'
    out = {k: base/k for k in ['figures','tables','video_review']}
    for p in out.values(): p.mkdir(parents=True, exist_ok=True)
    return out

def validate_inputs(root: Path | None = None) -> pd.DataFrame:
    root = root or repo_root(); rows=[]
    for name, rel in REQUIRED_INPUTS.items():
        p=root/rel; status=InputStatus(name, rel, p.exists())
        if p.exists() and p.suffix in {'.parquet','.csv'}:
            try:
                df = pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
                status=InputStatus(name, rel, True, len(df), len(df.columns))
            except Exception as e: status=InputStatus(name, rel, True, error=str(e))
        rows.append(status.__dict__)
    return pd.DataFrame(rows)

def read_optional_table(path: str | Path, root: Path | None = None) -> pd.DataFrame:
    p=(root or repo_root())/path
    if not p.exists(): return pd.DataFrame()
    return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)

def load_json_optional(path: str | Path, root: Path | None = None) -> dict:
    p=(root or repo_root())/path
    if not p.exists(): return {}
    return json.loads(p.read_text())

def join_oof(actions: pd.DataFrame, *preds: pd.DataFrame, keys: list[str] | None=None) -> pd.DataFrame:
    keys = keys or [k for k in EVENT_KEYS if k in actions.columns]
    out = actions.copy()
    if not keys: return out
    for i, pred in enumerate(preds):
        if pred is None or pred.empty or not set(keys).issubset(pred.columns): continue
        cols=[c for c in pred.columns if c in keys or c not in out.columns]
        out=out.merge(pred[cols].drop_duplicates(keys), on=keys, how='left', validate='m:1')
    return out

def validate_oof_alignment(actions: pd.DataFrame, preds: pd.DataFrame, keys: list[str] | None=None) -> dict:
    keys = keys or [k for k in EVENT_KEYS if k in actions.columns and k in preds.columns]
    if not keys: return {'has_keys': False, 'action_rows': len(actions), 'prediction_rows': len(preds), 'matched_rows': 0, 'duplicate_predictions': 0}
    a=actions[keys].drop_duplicates(); p=preds[keys]
    return {'has_keys': True, 'action_rows': len(actions), 'prediction_rows': len(preds), 'matched_rows': len(a.merge(p.drop_duplicates(), on=keys)), 'duplicate_predictions': int(p.duplicated(keys).sum())}
