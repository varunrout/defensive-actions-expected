from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from dax.analysis.schemas import validate_player_actions
from dax.analysis.player_aggregation import build_player_summary

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='data/features/player_defensive_actions.parquet'); p.add_argument('--output',default='data/features/player_defensive_summary.parquet'); p.add_argument('--min-actions',type=int,default=30); a=p.parse_args()
    df=pd.read_parquet(a.input); validate_player_actions(df); out=build_player_summary(df,a.min_actions); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_parquet(a.output,index=False); out.to_csv(Path(a.output).with_suffix('.csv'),index=False); print(f"Built player summary: {len(out)} players -> {a.output}")
if __name__=='__main__': main()
