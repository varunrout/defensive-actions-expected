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

SCHEMA_REQUIREMENTS = {
    'actions': {'any': [['match_id'], ['event_id']], 'recommended': ['competition', 'player', 'team']},
    'classification_oof': {'any': [['match_id', 'event_id']], 'recommended': ['fold']},
    'regression_oof': {'any': [['match_id', 'event_id']], 'recommended': ['fold']},
}


def validate_schema(df: pd.DataFrame, required_any: list[list[str]] | None = None, recommended: list[str] | None = None) -> dict:
    required_any = required_any or []
    recommended = recommended or []
    missing_options = [cols for cols in required_any if not set(cols).issubset(df.columns)]
    has_required = not required_any or len(missing_options) < len(required_any)
    missing_recommended = [col for col in recommended if col not in df.columns]
    return {
        'valid': bool(has_required),
        'columns': list(df.columns),
        'missing_required_options': missing_options if not has_required else [],
        'missing_recommended': missing_recommended,
    }


def select_variant(df: pd.DataFrame, variant: str, variant_cols: list[str] | None = None) -> pd.DataFrame:
    """Select a model variant explicitly when a variant column exists."""
    if df.empty:
        return df.copy()
    variant_cols = variant_cols or ['variant', 'model_variant', 'candidate', 'model_name', 'run_name']
    col = next((c for c in variant_cols if c in df.columns), None)
    if col is None:
        out = df.copy()
        out.attrs['variant_selection'] = f'no variant column; using all rows for requested {variant}'
        return out
    out = df[df[col].astype(str).eq(variant)].copy()
    out.attrs['variant_selection'] = f'{col} == {variant}'
    return out


def oof_coverage(actions: pd.DataFrame, preds: pd.DataFrame, keys: list[str] | None = None) -> dict:
    keys = keys or [k for k in EVENT_KEYS if k in actions.columns and k in preds.columns]
    alignment = validate_oof_alignment(actions, preds, keys)
    if not alignment.get('has_keys'):
        alignment.update({'missing_predictions': None, 'coverage_rate': None, 'match_coverage_rate': None, 'folds': []})
        return alignment
    action_keys = actions[keys].drop_duplicates()
    pred_keys = preds[keys].drop_duplicates()
    matched = action_keys.merge(pred_keys, on=keys, how='inner')
    action_matches = set(actions['match_id'].dropna()) if 'match_id' in actions.columns else set()
    pred_matches = set(preds['match_id'].dropna()) if 'match_id' in preds.columns else set()
    folds = sorted(preds['fold'].dropna().unique().tolist()) if 'fold' in preds.columns else []
    alignment.update({
        'missing_predictions': int(len(action_keys) - len(matched)),
        'coverage_rate': float(len(matched) / len(action_keys)) if len(action_keys) else None,
        'match_coverage_rate': float(len(action_matches & pred_matches) / len(action_matches)) if action_matches else None,
        'folds': folds,
        'fold_count': len(folds),
    })
    return alignment

class CoachAnalysisInputError(ValueError):
    """Raised when required coach-analysis inputs or variants are invalid."""


def require_table(path: str | Path, root: Path | None = None, required: bool = True) -> pd.DataFrame:
    p = Path(path)
    if not p.is_absolute():
        p = (root or repo_root()) / p
    if not p.exists():
        if required:
            raise CoachAnalysisInputError(f"Required input is missing: {p}")
        return pd.DataFrame()
    return pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)


def variant_columns_for(kind: str) -> list[str]:
    if kind == 'two_part':
        return ['classification_model_variant', 'conditional_model_variant']
    return ['model_variant', 'variant', 'candidate', 'model_name', 'run_name']


def available_variants(df: pd.DataFrame, kind: str = 'single') -> list[str] | list[tuple[str, str]]:
    if df.empty:
        return []
    if kind == 'two_part' and {'classification_model_variant', 'conditional_model_variant'}.issubset(df.columns):
        pairs = df[['classification_model_variant', 'conditional_model_variant']].drop_duplicates()
        return sorted([tuple(x) for x in pairs.to_numpy()])
    col = next((c for c in variant_columns_for(kind) if c in df.columns), None)
    if col is None:
        return ['<no variant column>'] if not df.empty else []
    return sorted(df[col].dropna().astype(str).unique().tolist())


def select_required_variant(df: pd.DataFrame, variant: str, kind: str = 'single', label: str = 'model') -> pd.DataFrame:
    if df.empty:
        raise CoachAnalysisInputError(f"No rows available for {label}; required variant {variant!r}.")
    col = next((c for c in variant_columns_for(kind) if c in df.columns), None)
    if col is None:
        raise CoachAnalysisInputError(f"No variant column found for {label}; available columns: {list(df.columns)}")
    selected = df[df[col].astype(str).eq(variant)].copy()
    if selected.empty:
        raise CoachAnalysisInputError(f"Required {label} variant {variant!r} unavailable. Available variants: {available_variants(df, kind)}")
    selected.attrs['variant_selection'] = f'{col} == {variant}'
    return selected


def select_required_two_part(df: pd.DataFrame, classification_variant: str, conditional_variant: str) -> pd.DataFrame:
    required = {'classification_model_variant', 'conditional_model_variant'}
    if df.empty:
        raise CoachAnalysisInputError('No rows available for two-part OOF predictions.')
    if not required.issubset(df.columns):
        raise CoachAnalysisInputError(f"Two-part OOF requires columns {sorted(required)}; available columns: {list(df.columns)}")
    mask = df['classification_model_variant'].astype(str).eq(classification_variant) & df['conditional_model_variant'].astype(str).eq(conditional_variant)
    selected = df[mask].copy()
    if selected.empty:
        raise CoachAnalysisInputError(
            f"Required two-part variant pair ({classification_variant!r}, {conditional_variant!r}) unavailable. "
            f"Available pairs: {available_variants(df, 'two_part')}"
        )
    selected.attrs['variant_selection'] = f'classification_model_variant == {classification_variant}; conditional_model_variant == {conditional_variant}'
    return selected


def validate_unique_predictions(preds: pd.DataFrame, keys: list[str] | None = None, label: str = 'predictions') -> None:
    keys = keys or [k for k in EVENT_KEYS if k in preds.columns]
    if not keys or not set(keys).issubset(preds.columns):
        raise CoachAnalysisInputError(f"{label} missing prediction keys {EVENT_KEYS}; available columns: {list(preds.columns)}")
    dupes = preds[preds.duplicated(keys, keep=False)]
    if not dupes.empty:
        examples = dupes[keys].head(10).to_dict('records')
        raise CoachAnalysisInputError(f"{label} contains duplicate predictions for {keys}; examples: {examples}")


def join_oof_strict(actions: pd.DataFrame, predictions: list[tuple[str, pd.DataFrame]], keys: list[str] | None = None) -> pd.DataFrame:
    keys = keys or [k for k in EVENT_KEYS if k in actions.columns]
    out = actions.copy()
    if not keys:
        raise CoachAnalysisInputError('Actions table has no match_id/event_id keys for OOF joining.')
    for label, pred in predictions:
        validate_unique_predictions(pred, keys, label)
        cols = [c for c in pred.columns if c in keys or c not in out.columns]
        out = out.merge(pred[cols], on=keys, how='left', validate='m:1')
    return out
