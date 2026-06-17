from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from dax.analysis.data_quality import processed_event_tables, write_tables
from dax.analysis.plotting import save_bar

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='data/processed/events_with_targets.parquet'); p.add_argument('--output-dir',default='outputs/analysis/data_quality'); p.add_argument('--config',default='configs/analysis.yaml'); a=p.parse_args()
    df=pd.read_parquet(a.input); tables=processed_event_tables(df); write_tables(tables,a.output_dir)
    od=Path(a.output_dir); save_bar(tables['event_counts_by_type'],'event_type','rows',od/'events_by_event_type.png','Events by event type'); save_bar(tables['rows_per_match'],'match_id','rows',od/'rows_per_match.png','Rows per match'); save_bar(tables['phase_distribution'],'phase_label','rows',od/'phase_distribution.png','Phase distribution')
    print(f"Analysed processed events: {len(df)} rows -> {a.output_dir}")
if __name__=='__main__': main()
