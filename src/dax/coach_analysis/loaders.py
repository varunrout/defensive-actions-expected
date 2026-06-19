from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

REQUIRED_INPUTS = {
    'actions': 'data/features/player_defensive_actions.parquet',
    'summary': 'data/features/player_defensive_summary.parquet',
    'signals': 'outputs/features/provisional_player_signals.parquet',
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

ACTION_REQUIRED_COLUMNS = (
    'match_id',
    'event_id',
    'period',
    'possession',
    'event_index',
    'phase_label',
    'competition_label',
    'team',
    'opponent_team',
    'player',
    'position',
    'position_group',
    'action_family',
    'event_type',
    'action_x',
    'action_y',
    'action_won_possession',
    'action_changed_possession',
    'has_360',
    'local_5m_region_fully_visible',
    'local_10m_region_fully_visible',
    'freeze_frame_roles_known',
    'visibility_quality_band',
    'target_future_shot_10s',
    'target_future_xg_10s',
)
CLASSIFICATION_REQUIRED_COLUMNS = ('match_id', 'event_id', 'fold', 'model_variant', 'y_true', 'y_score')
REGRESSION_REQUIRED_COLUMNS = ('match_id', 'event_id', 'fold', 'model_variant', 'y_true', 'y_pred')
TWO_PART_REQUIRED_COLUMNS = (
    'match_id',
    'event_id',
    'fold',
    'classification_model_variant',
    'conditional_model_variant',
    'observed_future_shot',
    'observed_future_xg',
    'combined_future_xg_prediction',
)


@dataclass(frozen=True)
class SchemaContract:
    name: str
    required: tuple[str, ...]


SCHEMA_CONTRACTS = {
    'actions': SchemaContract('player_defensive_actions', ACTION_REQUIRED_COLUMNS),
    'classification_oof': SchemaContract('classification_oof', CLASSIFICATION_REQUIRED_COLUMNS),
    'regression_oof': SchemaContract('regression_oof', REGRESSION_REQUIRED_COLUMNS),
    'two_part_oof': SchemaContract('two_part_oof', TWO_PART_REQUIRED_COLUMNS),
    'two_part_exploratory_oof': SchemaContract('two_part_exploratory_oof', TWO_PART_REQUIRED_COLUMNS),
}
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


def validate_schema_contract(df: pd.DataFrame, contract: SchemaContract) -> None:
    missing = [column for column in contract.required if column not in df.columns]
    if missing:
        raise ValueError(f"{contract.name} missing required columns: {missing}")


def schema_inventory(root: Path | None = None) -> dict[str, dict[str, object]]:
    root = root or repo_root()
    inventory: dict[str, dict[str, object]] = {}
    for key, rel in REQUIRED_INPUTS.items():
        path = root / rel
        if not path.exists() or path.suffix not in {'.parquet', '.csv'}:
            continue
        df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
        inventory[key] = {
            'path': rel,
            'rows': len(df),
            'columns': list(df.columns),
        }
    return inventory

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


def validate_unique_oof_predictions(
    predictions: pd.DataFrame,
    *,
    keys: tuple[str, str] = ('match_id', 'event_id'),
    prediction_column: str,
    low: float | None = None,
    high: float | None = None,
    name: str,
) -> None:
    for key in keys:
        if key not in predictions.columns:
            raise ValueError(f"{name} is missing key column '{key}'")
    duplicate_count = int(predictions.duplicated(list(keys)).sum())
    if duplicate_count:
        raise ValueError(f"{name} has duplicate rows for {list(keys)}: {duplicate_count}")
    if prediction_column not in predictions.columns:
        raise ValueError(f"{name} missing prediction column '{prediction_column}'")
    if predictions[prediction_column].isna().any():
        raise ValueError(f"{name} contains missing predictions in '{prediction_column}'")
    if low is not None and (predictions[prediction_column] < low).any():
        raise ValueError(f"{name} has values below {low} in '{prediction_column}'")
    if high is not None and (predictions[prediction_column] > high).any():
        raise ValueError(f"{name} has values above {high} in '{prediction_column}'")


def _select_variant(df: pd.DataFrame, variant_column: str, variant: str, *, name: str) -> pd.DataFrame:
    if variant_column not in df.columns:
        raise ValueError(f"{name} missing variant column '{variant_column}'")
    out = df[df[variant_column] == variant].copy()
    if out.empty:
        available = sorted(df[variant_column].dropna().astype(str).unique().tolist())
        raise ValueError(f"{name} variant '{variant}' unavailable. Available variants: {available}")
    return out


def select_classification_variant(df: pd.DataFrame, *, variant: str = 'b7_full_with_360') -> pd.DataFrame:
    validate_schema_contract(df, SCHEMA_CONTRACTS['classification_oof'])
    out = _select_variant(df, 'model_variant', variant, name='classification_oof')
    out = out.rename(
        columns={
            'y_score': 'coach_expected_shot_probability',
            'y_true': 'coach_observed_future_shot',
            'fold': 'coach_classification_fold',
            'model_variant': 'coach_classification_model_variant',
        }
    )
    validate_unique_oof_predictions(
        out,
        prediction_column='coach_expected_shot_probability',
        low=0.0,
        high=1.0,
        name=f'classification_oof[{variant}]',
    )
    return out


def select_regression_variant(df: pd.DataFrame, *, variant: str = 'r4_full_with_360') -> pd.DataFrame:
    validate_schema_contract(df, SCHEMA_CONTRACTS['regression_oof'])
    out = _select_variant(df, 'model_variant', variant, name='regression_oof')
    out = out.rename(
        columns={
            'y_pred': 'coach_expected_future_xg_r4',
            'y_true': 'coach_observed_future_xg',
            'fold': 'coach_regression_fold',
            'model_variant': 'coach_regression_model_variant',
        }
    )
    validate_unique_oof_predictions(
        out,
        prediction_column='coach_expected_future_xg_r4',
        low=0.0,
        name=f'regression_oof[{variant}]',
    )
    return out


def select_two_part_variant(
    df: pd.DataFrame,
    *,
    classification_variant: str = 'b7_full_with_360',
    conditional_variant: str = 'conditional_tweedie',
) -> pd.DataFrame:
    validate_schema_contract(df, SCHEMA_CONTRACTS['two_part_oof'])
    out = df[
        (df['classification_model_variant'] == classification_variant)
        & (df['conditional_model_variant'] == conditional_variant)
    ].copy()
    if out.empty:
        available = (
            df[['classification_model_variant', 'conditional_model_variant']]
            .drop_duplicates()
            .sort_values(['classification_model_variant', 'conditional_model_variant'])
        )
        raise ValueError(
            'two_part_oof variant '
            f"('{classification_variant}', '{conditional_variant}') unavailable. "
            f"Available pairs: {available.to_dict(orient='records')}"
        )
    out = out.rename(
        columns={
            'combined_future_xg_prediction': 'coach_expected_future_xg_two_part',
            'observed_future_xg': 'coach_observed_future_xg',
            'observed_future_shot': 'coach_observed_future_shot',
            'fold': 'coach_two_part_fold',
            'classification_model_variant': 'coach_two_part_classification_variant',
            'conditional_model_variant': 'coach_two_part_conditional_variant',
        }
    )
    validate_unique_oof_predictions(
        out,
        prediction_column='coach_expected_future_xg_two_part',
        low=0.0,
        name=f"two_part_oof[{classification_variant}+{conditional_variant}]",
    )
    return out


def validate_shared_row_coverage(
    actions: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    keys: tuple[str, str] = ('match_id', 'event_id'),
    prediction_column: str,
    name: str,
) -> dict[str, int]:
    action_keys = actions.loc[:, list(keys)].drop_duplicates()
    joined = action_keys.merge(predictions.loc[:, [*keys, prediction_column]], on=list(keys), how='left')
    missing = int(joined[prediction_column].isna().sum())
    if missing:
        raise ValueError(f"{name} missing predictions on eligible actions: {missing}")
    return {
        'eligible_actions': len(action_keys),
        'predicted_actions': int(joined[prediction_column].notna().sum()),
        'missing_predictions': missing,
    }


def _eligible_actions(actions: pd.DataFrame, *, requires_360: bool) -> pd.DataFrame:
    if not requires_360:
        return actions
    mask = actions.get('has_360', False).fillna(False).eq(True)
    if 'freeze_frame_roles_known' in actions.columns:
        mask = mask & actions['freeze_frame_roles_known'].fillna(False).eq(True)
    for feature in [
        'visible_attacker_count',
        'visible_defender_count',
        'attacker_defender_ratio',
        'nearest_attacker_distance',
        'nearest_defender_distance',
    ]:
        if feature in actions.columns:
            mask = mask & actions[feature].notna()
    return actions.loc[mask].copy()


def build_coach_analysis_frame(
    actions: pd.DataFrame,
    classification_oof: pd.DataFrame,
    regression_oof: pd.DataFrame,
    two_part_exploratory_oof: pd.DataFrame,
    *,
    classification_primary: str = 'b7_full_with_360',
    classification_sensitivity: str = 'b6_full_without_360',
    regression_primary: str = 'r4_full_with_360',
    two_part_classification_variant: str = 'b7_full_with_360',
    two_part_conditional_variant: str = 'conditional_tweedie',
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    validate_schema_contract(actions, SCHEMA_CONTRACTS['actions'])
    cls_primary = select_classification_variant(classification_oof, variant=classification_primary)
    cls_sensitivity = select_classification_variant(classification_oof, variant=classification_sensitivity).rename(
        columns={
            'coach_expected_shot_probability': 'coach_expected_shot_probability_sensitivity',
            'coach_classification_fold': 'coach_classification_fold_sensitivity',
            'coach_classification_model_variant': 'coach_classification_model_variant_sensitivity',
        }
    )
    reg_primary = select_regression_variant(regression_oof, variant=regression_primary)
    two_part = select_two_part_variant(
        two_part_exploratory_oof,
        classification_variant=two_part_classification_variant,
        conditional_variant=two_part_conditional_variant,
    )
    validations = {
        'classification_primary': validate_shared_row_coverage(
            _eligible_actions(actions, requires_360=True),
            cls_primary,
            prediction_column='coach_expected_shot_probability',
            name='classification_primary',
        ),
        'classification_sensitivity': validate_shared_row_coverage(
            _eligible_actions(actions, requires_360=False),
            cls_sensitivity,
            prediction_column='coach_expected_shot_probability_sensitivity',
            name='classification_sensitivity',
        ),
        'regression_primary': validate_shared_row_coverage(
            _eligible_actions(actions, requires_360=True),
            reg_primary,
            prediction_column='coach_expected_future_xg_r4',
            name='regression_primary',
        ),
        'two_part_exploratory': validate_shared_row_coverage(
            _eligible_actions(actions, requires_360=True),
            two_part,
            prediction_column='coach_expected_future_xg_two_part',
            name='two_part_exploratory',
        ),
    }

    merged = actions.copy()
    for frame in (cls_primary, cls_sensitivity, reg_primary, two_part):
        keep_cols = [
            column
            for column in frame.columns
            if column in EVENT_KEYS or (column.startswith('coach_') and column not in merged.columns)
        ]
        merged = merged.merge(frame[keep_cols], on=EVENT_KEYS, how='left', validate='1:1')
    return merged, validations

def validate_oof_alignment(actions: pd.DataFrame, preds: pd.DataFrame, keys: list[str] | None=None) -> dict:
    keys = keys or [k for k in EVENT_KEYS if k in actions.columns and k in preds.columns]
    if not keys: return {'has_keys': False, 'action_rows': len(actions), 'prediction_rows': len(preds), 'matched_rows': 0, 'duplicate_predictions': 0}
    a=actions[keys].drop_duplicates(); p=preds[keys]
    return {'has_keys': True, 'action_rows': len(actions), 'prediction_rows': len(preds), 'matched_rows': len(a.merge(p.drop_duplicates(), on=keys)), 'duplicate_predictions': int(p.duplicated(keys).sum())}
