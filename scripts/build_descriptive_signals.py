from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from dax.analysis.signal_design import build_descriptive_signals

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='data/features/player_defensive_summary.parquet'); p.add_argument('--clusters',default='outputs/analysis/clustering/player_clusters.parquet'); p.add_argument('--output',default='data/features/player_defensive_signals_descriptive.parquet'); p.add_argument('--min-actions',type=int,default=30); a=p.parse_args()
    df=pd.read_parquet(a.input); clusters=pd.read_parquet(a.clusters) if Path(a.clusters).exists() else None; out=build_descriptive_signals(df,clusters,a.min_actions); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_parquet(a.output,index=False); out.to_csv(Path(a.output).with_suffix('.csv'),index=False); print(f"Built descriptive signals: {len(out)} players -> {a.output}")
if __name__=='__main__': main()
