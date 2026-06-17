from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from dax.analysis.schemas import validate_player_actions
from dax.analysis.data_quality import write_tables, missingness_summary
from dax.analysis.feature_diagnostics import diagnostics_tables
from dax.analysis.spatial_analysis import zone_summary, player_spatial_profiles
from dax.analysis.phase_analysis import phase_tables
from dax.analysis.plotting import save_bar, save_heatmap

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='data/features/player_defensive_actions.parquet'); p.add_argument('--output-dir',default='outputs/analysis/features'); p.add_argument('--config',default='configs/analysis.yaml'); a=p.parse_args()
    df=pd.read_parquet(a.input); validate_player_actions(df); tables=diagnostics_tables(df); tables['missingness']=missingness_summary(df); tables['zone_summary']=zone_summary(df); tables['player_spatial_profiles']=player_spatial_profiles(df); tables.update(phase_tables(df)); write_tables(tables,a.output_dir)
    od=Path(a.output_dir); save_bar(df['action_family'].value_counts().rename_axis('action_family').reset_index(name='rows'),'action_family','rows',od/'action_family_distribution.png','Action-family distribution'); save_heatmap(tables['correlations'],od/'correlation_heatmap.png','Feature correlation heatmap')
    print(f"Analysed player defensive-action features: {len(df)} rows -> {a.output_dir}")
if __name__=='__main__': main()
