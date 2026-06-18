from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from .schemas import normalise_model_schema, validate_model_dataset, dataset_fingerprint
from .feature_contracts import load_model_config, get_contracts, resolve_contract
from .splits import make_grouped_folds, fold_summary
from .classification import build_classifier
from .regression import build_regressor
from .evaluation import classification_metrics, regression_metrics
from .mlflow_tracking import configure_mlflow,start_parent_run,start_variant_run,log_params,log_metrics,log_json_artifact,log_dataset_fingerprint

def _git_sha():
    import subprocess
    try: return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception: return None

def run_training(task,input_path,config_path='configs/models.yaml',output_dir='outputs',mlflow_enabled=None,tracking_uri=None,n_splits=None,max_rows=None):
    cfg=load_model_config(config_path); df=normalise_model_schema(pd.read_parquet(input_path))
    if max_rows: df=df.head(max_rows).copy()
    audit=validate_model_dataset(df); fp=dataset_fingerprint(input_path,df); folds=make_grouped_folds(df,cfg[task]['target'],cfg.get('group_column','match_id'),n_splits or cfg.get('folds',5),cfg.get('seed',42))
    out=Path(output_dir); (out/'models/splits').mkdir(parents=True,exist_ok=True); (out/'oof').mkdir(parents=True,exist_ok=True); (out/'models/comparisons').mkdir(parents=True,exist_ok=True); (out/f'models/{task}').mkdir(parents=True,exist_ok=True)
    folds.to_parquet(out/'models/splits/fold_assignments.parquet',index=False); fold_summary(df,cfg[task]['target'],folds).to_csv(out/f'models/{task}/fold_summary.csv',index=False)
    ml=configure_mlflow(cfg, False if mlflow_enabled is None else mlflow_enabled, tracking_uri); exp=cfg['mlflow'][f'{task}_experiment'] if task in {'classification','regression'} else 'dax-models'
    all_oof=[]; comparisons=[]
    with start_parent_run(ml,exp,f"{cfg['mlflow'].get('run_name_prefix','baseline')}-{task}") as pr:
      parent_id=getattr(getattr(pr,'info',None),'run_id',None); log_dataset_fingerprint(ml,fp)
      for c in get_contracts(cfg,task):
        try: resolved=resolve_contract(df,c)
        except ValueError as e: comparisons.append({'variant':c.name,'model_family':c.model_family,'recommendation_status':f'skipped: {e}'}); continue
        if audit.rows < c.minimum_usable_rows: continue
        with start_variant_run(ml,c.name) as vr:
          run_id=getattr(getattr(vr,'info',None),'run_id',None); t0=time.perf_counter(); preds=np.full(len(df),np.nan); train_matches_by_fold={}
          for f in sorted(folds.fold.unique()):
            val=folds.fold.eq(f).to_numpy(); tr=~val; train_matches_by_fold[int(f)]=sorted(map(str,df.loc[tr,'match_id'].unique()))
            Xtr=df.loc[tr,resolved['final_features']] if resolved['final_features'] else pd.DataFrame(index=df.index[tr])
            Xv=df.loc[val,resolved['final_features']] if resolved['final_features'] else pd.DataFrame(index=df.index[val])
            ytr=df.loc[tr,c.target]
            model=build_classifier(c,resolved) if task=='classification' else build_regressor(c,resolved); model.fit(Xtr,ytr)
            preds[val]=model.predict_proba(Xv)[:,1] if task=='classification' else model.predict(Xv)
          if task=='regression' and cfg['regression'].get('clip_negative_predictions',False): preds=np.clip(preds,0,None)
          metrics=classification_metrics(df[c.target],preds) if task=='classification' else regression_metrics(df[c.target],preds)
          final_model=build_classifier(c,resolved) if task=='classification' else build_regressor(c,resolved); X=df[resolved['final_features']] if resolved['final_features'] else pd.DataFrame(index=df.index); final_model.fit(X,df[c.target])
          bundle={'pipeline':final_model,'target':c.target,'feature_contract':c.__dict__,'final_feature_list':resolved['final_features'],'preprocessing_metadata':resolved,'training_rows':len(df),'training_matches':audit.matches,'data_fingerprint':fp,'model_version':'0.1.0','git_commit_sha':_git_sha(),'mlflow_run_id':run_id,'timestamp':datetime.now(timezone.utc).isoformat()}
          mp=out/f'models/{task}/{c.name}.joblib'; joblib.dump(bundle,mp)
          ident=[col for col in ['event_id','match_id','player_id','player_name','team','action_family','phase_label','position_group'] if col in df.columns]
          oof=df[ident].copy(); oof['y_true']=df[c.target].to_numpy(); oof['fold']=folds.fold.to_numpy(); oof['model_variant']=c.name; oof['model_family']=c.model_family; oof['mlflow_run_id']=run_id; oof['train_match_ids']=oof['fold'].map(train_matches_by_fold)
          if task=='classification': oof['y_score']=preds
          else: oof['y_pred']=preds; oof['residual']=oof['y_pred']-oof['y_true']
          all_oof.append(oof); elapsed=time.perf_counter()-t0
          comparisons.append({'variant':c.name,'model_family':c.model_family,'feature_count':len(resolved['final_features']),'rows':len(df),'matches':audit.matches,'training_time':elapsed,'inference_time':0.0,'requires_360':c.requires_360,'missing_features':';'.join(resolved['missing_optional_features']),'mlflow_run_id':run_id,'recommendation_status':'candidate',**metrics})
          log_params(ml,{'target':c.target,'model_family':c.model_family,'model_variant':c.name,'feature_scope':c.feature_scope,'features':resolved['final_features'],'fold_count':folds.fold.nunique(),'seed':cfg.get('seed',42),'rows':len(df),'matches':audit.matches,'target_prevalence':audit.shot_rate,'coverage_360':resolved['coverage_360'],'hyperparameters':c.hyperparameters,'git_sha':_git_sha(),'data_fingerprint':fp}); log_metrics(ml,metrics); log_json_artifact(ml,{'contract':c.__dict__,'resolved':resolved},out/f'models/{task}/{c.name}_feature_contract.json')
    if all_oof:
      oof_all=pd.concat(all_oof,ignore_index=True); kind='classification' if task=='classification' else 'regression'; oof_all.to_parquet(out/f'oof/{kind}_oof.parquet',index=False)
    comp=pd.DataFrame(comparisons); comp.to_csv(out/f'models/comparisons/{task}_model_comparison.csv',index=False)
    return {'parent_run_id':parent_id,'comparison':str(out/f'models/comparisons/{task}_model_comparison.csv'),'comparison_frame':comp}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--task',choices=['classification','regression','logistic','all'],required=True); p.add_argument('--input',default='data/features/player_defensive_actions.parquet'); p.add_argument('--config',default='configs/models.yaml'); p.add_argument('--output-dir',default='outputs'); p.add_argument('--models-dir'); p.add_argument('--validation-dir'); p.add_argument('--oof-dir'); p.add_argument('--mlflow-enabled',action=argparse.BooleanOptionalAction,default=None); p.add_argument('--tracking-uri'); p.add_argument('--n-splits',type=int); p.add_argument('--max-rows',type=int); a=p.parse_args(argv) 
    tasks=['classification','regression'] if a.task=='all' else [('classification' if a.task=='logistic' else a.task)]
    # Backward-compatible legacy directory flags are accepted; canonical output-dir is used.
    for t in tasks:
        res=run_training(t,a.input,a.config,a.output_dir,a.mlflow_enabled,a.tracking_uri,a.n_splits,a.max_rows)
        print(res)
        if a.validation_dir:
            import json as _json
            from pathlib import Path as _Path
            legacy='baseline' if t=='classification' else 'regression'
            d=_Path(a.validation_dir)/legacy; d.mkdir(parents=True,exist_ok=True)
            rows=res['comparison_frame'].to_dict(orient='records') if hasattr(res['comparison_frame'],'to_dict') else []
            variants={str(row.get('variant')):{**row, 'avg_precision': row.get('average_precision', row.get('avg_precision'))} for row in rows}
            summary={'task':t,'rows': int(rows[0].get('rows',0)) if rows else 0,'matches': int(rows[0].get('matches',0)) if rows else 0,'target_rate': float(rows[0].get('positive_rate',0.0)) if rows else 0.0,'target_mean': float(rows[0].get('mean_observed',0.0)) if rows else 0.0,'variants':variants}
            (d/('baseline_model_metrics.json' if t=='classification' else 'regression_model_metrics.json')).write_text(_json.dumps(summary,indent=2,default=str))
            res['comparison_frame'].to_csv(d/('baseline_model_metrics_table.csv' if t=='classification' else 'regression_model_metrics_table.csv'),index=False)
        if a.oof_dir:
            from pathlib import Path as _Path
            import shutil as _shutil
            src=_Path(a.output_dir)/'oof'/('classification_oof.parquet' if t=='classification' else 'regression_oof.parquet')
            _Path(a.oof_dir).mkdir(parents=True,exist_ok=True)
            if src.exists(): _shutil.copy2(src, _Path(a.oof_dir)/src.name)
if __name__=='__main__': main()
