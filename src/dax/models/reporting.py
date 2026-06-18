from __future__ import annotations
from pathlib import Path
import pandas as pd

def write_model_report(config_path='configs/models.yaml', output='outputs/models/reports/model_validation_report.md', outputs_dir='outputs'):
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); lines=['# Model validation report','','Provisional predictive modelling report. No output is presented as true DAx or causal.']
    for task in ['classification','regression']:
        p=Path(outputs_dir)/'models/comparisons'/f'{task}_model_comparison.csv'
        if p.exists(): lines += ['',f'## {task.title()} comparison','',pd.read_csv(p).to_csv(index=False)]
    out.write_text('\n'.join(lines)); return out
