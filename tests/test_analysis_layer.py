from __future__ import annotations
import pandas as pd
from dax.analysis.schemas import validate_player_actions, validate_processed_events
from dax.analysis.data_quality import missingness_summary, processed_event_tables
from dax.analysis.spatial_analysis import add_pitch_zones, zone_summary
from dax.analysis.player_aggregation import build_player_summary
from dax.analysis.clustering import prepare_clustering_matrix, run_clustering
from dax.analysis.signal_design import build_descriptive_signals
from dax.analysis.reporting import generate_pre_model_report
from dax.analysis.plotting import save_bar

def player_df(n=60):
    rows=[]
    for i in range(n):
        rows.append({"match_id":i//10,"event_id":i,"player_id":i%6,"player_name":f"P{i%6}","team":f"T{i%2}","actor_team":f"T{i%2}","attacking_team":f"T{(i+1)%2}","defending_team":f"T{i%2}","event_type":"Duel" if i%2 else "Interception","action_family":"duel" if i%2 else "interception","phase_label":"counterpress" if i%3 else "low_block","x":float(i%120),"y":float(i%80),"has_360":i%2==0,"visible_attacker_count":i%5,"visible_defender_count":i%4,"local_numerical_balance":(i%5)-2,"nearest_attacker_distance":float(i%10),"nearest_defender_distance":float(i%8),"goal_side_defenders":i%3,"possession_won":i%4==0,"ends_opponent_possession":i%5==0,"retained_control":i%6==0,"under_opponent_possession":True,"target_future_shot_10s":i%7==0,"target_future_xg_10s":0.1 if i%7==0 else 0.0})
    return pd.DataFrame(rows)

def event_df():
    df=player_df(40).rename(columns={"attacking_team":"attacking_team_before_action","defending_team":"defending_team_before_action"})
    df["period"]=1; df["index"]=range(len(df)); df["minute"]=0; df["possession"]=df.index//4; return df

def test_schema_and_quality():
    assert validate_processed_events(event_df()).ok
    assert validate_player_actions(player_df()).ok
    assert not missingness_summary(player_df()).empty
    assert "overview" in processed_event_tables(event_df())

def test_spatial_and_player_aggregation_denominators():
    df=player_df(); assert "pitch_zone" in add_pitch_zones(df)
    assert not zone_summary(df).empty
    s=build_player_summary(df,min_actions=5)
    assert (s["future_shot_denominator"]==s["total_actions"]).all()
    assert "minimum_sample_flag" in s

def test_clustering_and_signals():
    s=build_player_summary(player_df(120),min_actions=5)
    matrix, audit, meta=prepare_clustering_matrix(s,min_actions=5)
    assert meta["features"]
    tables=run_clustering(matrix,candidates=[2,3],seed=1)
    assert not tables["cluster_evaluation"].empty
    sig=build_descriptive_signals(s,tables["player_clusters"],min_actions=5)
    assert "activity_index" in sig

def test_plot_and_report(tmp_path):
    save_bar(pd.DataFrame({"x":["a"],"y":[1]}),"x","y",tmp_path/"bar.png","Bar")
    assert (tmp_path/"bar.png").exists()
    dq=tmp_path/"data_quality"; dq.mkdir()
    pd.DataFrame([{"rows":10,"future_shot_rate":0.1}]).to_csv(dq/"overview.csv",index=False)
    pd.DataFrame([{"duplicate_rows":0}]).to_csv(dq/"duplicates.csv",index=False)
    pd.DataFrame([{"column":"a","missing_rate":0.0}]).to_csv(dq/"missingness.csv",index=False)
    c=tmp_path/"clustering"; c.mkdir(); pd.DataFrame([{"method":"kmeans"}]).to_csv(c/"cluster_evaluation.csv",index=False)
    out=generate_pre_model_report(tmp_path,tmp_path/"reports"/"report.md")
    assert out.exists()
