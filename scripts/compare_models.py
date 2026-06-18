import argparse
from pathlib import Path
import pandas as pd

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/models.yaml'); p.add_argument('--output-dir',default='outputs'); p.add_argument('--mlflow-enabled',action='store_true'); p.add_argument('--tracking-uri'); a=p.parse_args(argv)
 for task in ['classification','regression']:
  path=Path(a.output_dir)/'models/comparisons'/f'{task}_model_comparison.csv'
  if path.exists():
   df=pd.read_csv(path); df['selection_note']='transparent multi-metric review required; no single-metric winner'; df.to_csv(path,index=False); print(path)
if __name__=='__main__': main()
