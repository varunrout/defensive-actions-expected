from __future__ import annotations
from pathlib import Path
import pandas as pd

def assert_oof_predictions(oof:pd.DataFrame, score_col:str):
    keys=['event_id','model_variant'] if 'event_id' in oof else ['match_id','player_id','model_variant']
    if oof.duplicated(keys).any(): raise ValueError('Duplicate OOF event-variant predictions detected.')
    if oof[score_col].isna().any(): raise ValueError('Missing OOF predictions detected.')
    if 'train_match_ids' in oof:
        bad=oof.apply(lambda r: str(r['match_id']) in set(map(str,r['train_match_ids'])),axis=1)
        if bad.any(): raise ValueError('OOF fold isolation violated: validation match in training matches.')
    return True

def validate_outputs(task:str, output_dir='outputs'):
    p=Path(output_dir)/'oof'/('classification_oof.parquet' if task=='classification' else 'regression_oof.parquet')
    if not p.exists(): raise FileNotFoundError(p)
    oof=pd.read_parquet(p); assert_oof_predictions(oof,'y_score' if task=='classification' else 'y_pred'); return {'rows':len(oof),'variants':int(oof.model_variant.nunique())}
