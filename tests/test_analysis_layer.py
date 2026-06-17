from __future__ import annotations

from pathlib import Path

import pandas as pd

from dax.analysis.clustering import prepare_clustering_matrix, run_clustering
from dax.analysis.config import load_analysis_config
from dax.analysis.data_quality import missingness_summary, processed_event_tables
from dax.analysis.player_aggregation import build_player_summary
from dax.analysis.plotting import bar_chart
from dax.analysis.reporting import generate_pre_model_report
from dax.analysis.schemas import validate_player_actions, validate_processed_events
from dax.analysis.signal_design import build_descriptive_signals
from dax.analysis.spatial_analysis import add_pitch_zones, zone_summary
from dax.features.player_defense import build_player_defensive_actions

FULL_VISIBLE_AREA = [0, 0, 120, 0, 120, 80, 0, 80]
SMALL_VISIBLE_AREA = [40, 20, 80, 20, 80, 60, 40, 60]
EVENT_TYPES = ["Pressure", "Ball Recovery", "Duel", "Interception", "Clearance", "Block"]
PHASES = ["high_press", "counterpress", "transition_defence", "low_block"]


def production_player_actions_fixture(players: int = 12, actions_per_player: int = 4) -> pd.DataFrame:
    """Build analysis test data through the production player feature builder."""
    events: list[dict[str, object]] = []
    index = 1
    for player_number in range(players):
        for action_number in range(actions_per_player):
            team = "Team B" if player_number % 2 == 0 else "Team A"
            opponent = "Team A" if team == "Team B" else "Team B"
            event_type = EVENT_TYPES[(player_number + action_number) % len(EVENT_TYPES)]
            x = 30.0 + ((player_number * 7 + action_number * 11) % 75)
            y = 15.0 + ((player_number * 5 + action_number * 9) % 50)
            visible_area = FULL_VISIBLE_AREA if action_number % 3 else SMALL_VISIBLE_AREA
            events.append(
                {
                    "match_id": 1 + player_number // 6,
                    "period": 1,
                    "possession": index,
                    "index": index,
                    "minute": index // 60,
                    "second": index % 60,
                    "id": f"event-{index}",
                    "event_type": event_type,
                    "player_id": 1000 + player_number,
                    "player": f"Player {player_number}",
                    "position": "Centre Back" if player_number % 3 == 0 else "Midfield",
                    "team": team,
                    "actor_team": team,
                    "attacking_team_before_action": opponent,
                    "defending_team_before_action": team,
                    "location": [x, y],
                    "ball_x": x,
                    "ball_y": y,
                    "has_360": True,
                    "freeze_frame_count": 4,
                    "freeze_frame": [
                        {"teammate": True, "location": [max(0.0, x - 5), y]},
                        {"teammate": True, "location": [min(120.0, x + 5), y]},
                        {"teammate": False, "location": [min(120.0, x + 10), y + 3]},
                        {"teammate": False, "location": [min(120.0, x + 15), y - 3]},
                    ],
                    "visible_area": visible_area,
                    "phase_label": PHASES[(player_number + action_number) % len(PHASES)],
                    "play_pattern": "Regular Play",
                    "counterpress": action_number % 2 == 0,
                    "action_changed_possession": event_type in {"Ball Recovery", "Interception"},
                    "action_ended_possession": event_type in {"Ball Recovery", "Interception", "Clearance"},
                    "action_won_possession": event_type in {"Ball Recovery", "Interception"},
                    "action_retained_defensive_team_control": event_type in {"Ball Recovery", "Interception"},
                    "action_was_under_opponent_possession": True,
                    "target_future_shot_10s": int((player_number + action_number) % 5 == 0),
                    "target_future_xg_10s": 0.08 if (player_number + action_number) % 5 == 0 else 0.0,
                }
            )
            index += 1
    rows = build_player_defensive_actions(events, only_with_360=True, defensive_only=True)
    return pd.DataFrame(rows)


def processed_events_fixture() -> pd.DataFrame:
    players = production_player_actions_fixture(players=4, actions_per_player=3)
    events = players.rename(
        columns={
            "event_index": "index",
            "attacking_team": "attacking_team_before_action",
            "defending_team": "defending_team_before_action",
        }
    ).copy()
    events["minute"] = 0
    return events


def test_schema_uses_production_player_feature_contract() -> None:
    df = production_player_actions_fixture()
    assert validate_player_actions(df).ok
    forbidden = {
        "local_numerical_balance",
        "goal_side_defenders",
        "visibility_quality",
        "possession_won",
        "ends_opponent_possession",
        "retained_control",
        "under_opponent_possession",
    }
    assert forbidden.isdisjoint(df.columns)


def test_processed_quality_and_spatial_tables() -> None:
    events = processed_events_fixture()
    assert validate_processed_events(events).ok
    assert not missingness_summary(events).empty
    assert "overview" in processed_event_tables(events)
    actions = production_player_actions_fixture()
    assert "pitch_zone" in add_pitch_zones(actions)
    assert not zone_summary(actions).empty


def test_production_fixture_full_analysis_chain(tmp_path: Path) -> None:
    config = load_analysis_config(None)
    config["minimum_player_actions"] = 2
    config["cluster_count_candidates"] = [2, 3]
    actions = production_player_actions_fixture(players=12, actions_per_player=4)
    validate_player_actions(actions)

    summary = build_player_summary(actions, min_actions=2, grid_dimensions=tuple(config["pitch_grid_dimensions"]))
    assert {"role_known_actions", "reliable_visibility_actions", "numerical_disadvantage_10m_denominator"}.issubset(summary.columns)
    assert (summary["future_shot_denominator"] == summary["total_actions"]).all()

    matrix, audit, metadata = prepare_clustering_matrix(summary, config)
    assert len(matrix) == 12
    assert metadata["selected_features"]
    assert not any(feature.startswith("future_") for feature in metadata["selected_features"])

    tables = run_clustering(matrix, config)
    assert {"kmeans", "hierarchical", "gmm"}.issubset(set(tables["cluster_evaluation"]["method"]))
    assert tables["cluster_stability"]["subsample_ari_stability"].notna().any()

    signals = build_descriptive_signals(summary, tables["player_clusters"], min_actions=2)
    assert "activity_index" in signals
    assert signals["warnings"].str.contains("not true DAx").all()

    analysis_dir = tmp_path / "analysis"
    data_quality_dir = analysis_dir / "data_quality"
    clustering_dir = analysis_dir / "clustering"
    data_quality_dir.mkdir(parents=True)
    clustering_dir.mkdir(parents=True)
    processed_event_tables(processed_events_fixture())["overview"].to_csv(data_quality_dir / "overview.csv", index=False)
    processed_event_tables(processed_events_fixture())["duplicates"].to_csv(data_quality_dir / "duplicates.csv", index=False)
    processed_event_tables(processed_events_fixture())["missingness"].to_csv(data_quality_dir / "missingness.csv", index=False)
    tables["cluster_evaluation"].to_csv(clustering_dir / "cluster_evaluation.csv", index=False)
    report = generate_pre_model_report(analysis_dir, analysis_dir / "reports" / "report.md")
    assert report.exists()


def test_plotting_handles_empty_and_non_empty_data(tmp_path: Path) -> None:
    bar_chart(pd.DataFrame({"x": ["a"], "y": [1]}), "x", "y", tmp_path / "bar.png", "Bar")
    bar_chart(pd.DataFrame(columns=["x", "y"]), "x", "y", tmp_path / "empty.png", "Empty")
    assert (tmp_path / "bar.png").exists()
    assert (tmp_path / "empty.png").exists()
