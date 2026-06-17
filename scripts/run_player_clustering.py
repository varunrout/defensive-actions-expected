from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from dax.analysis.clustering import prepare_clustering_matrix, run_clustering, write_clustering_outputs

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='data/features/player_defensive_summary.parquet'); p.add_argument('--output-dir',default='outputs/analysis/clustering'); p.add_argument('--config',default='configs/analysis.yaml'); p.add_argument('--matrix-output',default='data/features/player_clustering_matrix.parquet'); a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()) if Path(a.config).exists() else {}; df=pd.read_parquet(a.input); matrix,audit,meta=prepare_clustering_matrix(df,cfg.get('minimum_player_actions',30)); Path(a.output_dir).mkdir(parents=True,exist_ok=True); Path(a.matrix_output).parent.mkdir(parents=True,exist_ok=True); matrix.to_parquet(a.matrix_output,index=False); audit.to_csv(Path(a.output_dir)/'clustering_unscaled_audit.csv',index=False); tables=run_clustering(matrix,cfg.get('cluster_count_candidates',[2,3,4]),cfg.get('random_seed',42)); write_clustering_outputs(tables,a.output_dir,meta); print(f"Ran clustering for {len(matrix)} players -> {a.output_dir}")
if __name__=='__main__': main()
