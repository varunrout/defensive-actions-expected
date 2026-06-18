from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import json
import os
class DisabledRun:
    info=type('Info',(),{'run_id':None})()
    def __enter__(self): return self
    def __exit__(self,*a): return False

def _mlflow():
    try: import mlflow; return mlflow
    except Exception as e: raise RuntimeError('MLflow is required when mlflow.enabled=true') from e

def configure_mlflow(cfg, enabled=None, tracking_uri=None):
    mcfg=cfg.get('mlflow',{}) if cfg else {}; en=mcfg.get('enabled',True) if enabled is None else enabled
    if not en: return None
    ml=_mlflow(); uri=tracking_uri or os.getenv('MLFLOW_TRACKING_URI') or mcfg.get('tracking_uri')
    if uri: ml.set_tracking_uri(uri)
    return ml

def start_parent_run(ml, experiment, run_name):
    if ml is None: return DisabledRun()
    ml.set_experiment(experiment); return ml.start_run(run_name=run_name)
def start_variant_run(ml, run_name):
    return DisabledRun() if ml is None else ml.start_run(run_name=run_name,nested=True)
def log_params(ml, params):
    if ml: 
        for k,v in params.items(): ml.log_param(k, json.dumps(v)[:500] if isinstance(v,(list,dict)) else v)
def log_metrics(ml, metrics, prefix=''):
    if ml:
        for k,v in metrics.items():
            try: ml.log_metric(prefix+k, float(v))
            except Exception: pass
def log_json_artifact(ml,obj,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,indent=2,default=str))
    if ml: ml.log_artifact(str(p))
def log_dataset_fingerprint(ml, fp, path='outputs/models/reports/dataset_fingerprint.json'): log_json_artifact(ml,fp,path)
def log_feature_contract(ml, contract, resolved, path): log_json_artifact(ml,{'contract':contract.__dict__,'resolved':resolved},path)
def log_fold_metrics(ml, path):
    if ml and Path(path).exists(): ml.log_artifact(str(path))
def log_classification_metrics(ml, metrics): log_metrics(ml,metrics)
def log_regression_metrics(ml, metrics): log_metrics(ml,metrics)
def log_model_artifacts(ml, paths):
    if ml:
        for p in paths:
            if Path(p).exists(): ml.log_artifact(str(p))
def log_chart_directory(ml, directory):
    if ml and Path(directory).exists(): ml.log_artifacts(str(directory))
