import argparse
from pathlib import Path
import pandas as pd
from dax.models.validation import validate_outputs

def _legacy_table(validation_dir, task):
    legacy = "baseline" if task == "classification" else "regression"
    d = Path(validation_dir) / legacy
    src = d / ("baseline_model_metrics.json" if task == "classification" else "regression_model_metrics.json")
    dst = d / ("baseline_model_metrics_table.csv" if task == "classification" else "regression_model_metrics_table.csv")
    if src.exists() and not dst.exists():
        import json
        data = json.loads(src.read_text())
        pd.DataFrame(data.get("variants", [])).to_csv(dst, index=False)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--task',choices=['classification','regression','logistic','all'],required=True); p.add_argument('--config',default='configs/models.yaml'); p.add_argument('--output-dir',default='outputs'); p.add_argument('--validation-dir'); p.add_argument('--oof-dir'); p.add_argument('--mlflow-enabled',action='store_true'); p.add_argument('--tracking-uri'); a=p.parse_args(argv)
    tasks=['classification','regression'] if a.task=='all' else [('classification' if a.task=='logistic' else a.task)]
    for t in tasks:
        if a.validation_dir: _legacy_table(a.validation_dir,t); print({'task':t,'validation_dir':a.validation_dir})
        else: print(validate_outputs(t,a.output_dir))
if __name__=='__main__': main()
