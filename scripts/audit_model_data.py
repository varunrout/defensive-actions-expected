from __future__ import annotations
import argparse
import json
import pandas as pd
from dax.models.schemas import normalise_model_schema, validate_model_dataset, dataset_fingerprint
from dax.models.feature_contracts import load_model_config, get_contracts, resolve_contract

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--config',default='configs/models.yaml'); p.add_argument('--output',default='outputs/models/reports/model_data_audit.json'); a=p.parse_args(argv)
 df=normalise_model_schema(pd.read_parquet(a.input)); audit=validate_model_dataset(df); cfg=load_model_config(a.config); variants={}
 for task in ['classification','regression']:
  variants[task]={}
  for c in get_contracts(cfg,task):
   try: variants[task][c.name]=resolve_contract(df,c)
   except Exception as e: variants[task][c.name]={'error':str(e)}
 import pathlib; pathlib.Path(a.output).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(a.output).write_text(json.dumps({'audit':audit.__dict__,'fingerprint':dataset_fingerprint(a.input,df),'variants':variants},indent=2,default=str)); print(audit)
if __name__=='__main__': main()
