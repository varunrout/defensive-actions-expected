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
    """Return a production-shaped processed event fixture with `type`, not `event_type`."""
    rows = []
    event_types = ["Pass", "Pressure", "Shot", "Ball Recovery", "Interception", "Duel"]
    for index, event_type in enumerate(event_types, start=1):
        rows.append(
            {
                "match_id": 1,
                "period": 1,
                "index": index,
                "minute": 0,
                "second": index * 5,
                "event_time_seconds": index * 5.0,
                "possession": 1 + index // 3,
                "id": f"processed-{index}",
                "type": event_type,
                "duel_type": None,
                "goalkeeper_type": None,
                "pass_type": "Regular" if event_type == "Pass" else None,
                "related_events": [],
                "shot_type": "Open Play" if event_type == "Shot" else None,
                "foul_committed_type": None,
                "previous_event_team": "Team A" if index > 1 else None,
                "event_starts_new_possession": index in {1, 3},
                "event_semantics_known": True,
                "team": "Team A" if index in {1, 3} else "Team B",
                "player": f"Event Player {index}",
                "phase_label": PHASES[index % len(PHASES)],
                "has_360": True,
                "attacking_team_before_action": "Team A",
                "defending_team_before_action": "Team B",
                "target_future_shot_10s": int(event_type == "Pressure"),
                "target_future_xg_10s": 0.12 if event_type == "Pressure" else 0.0,
            }
        )
    return pd.DataFrame(rows)


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


def test_processed_schema_requires_type_not_event_type(tmp_path: Path) -> None:
    import subprocess
    import sys

    events = processed_events_fixture()
    assert validate_processed_events(events).ok
    with_missing_type = events.drop(columns=["type"])
    try:
        validate_processed_events(with_missing_type)
    except ValueError as exc:
        assert "type" in str(exc)
    else:
        raise AssertionError("processed event validation should reject fixtures missing `type`")

    tables = processed_event_tables(events)
    assert "type" in tables["event_counts_by_type"].columns
    assert "event_type" not in tables["event_counts_by_type"].columns

    input_path = tmp_path / "events.parquet"
    output_dir = tmp_path / "data_quality"
    events.to_parquet(input_path, index=False)
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_processed_data.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--config",
            "configs/analysis.yaml",
        ],
        check=True,
    )
    counts = pd.read_csv(output_dir / "event_counts_by_type.csv")
    assert "type" in counts.columns


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


def test_defensive_box_flag_uses_penalty_box_boundaries() -> None:
    rows = []
    base = production_player_actions_fixture(players=1, actions_per_player=1).iloc[0].to_dict()
    locations = [(10.0, 40.0), (10.0, 70.0), (30.0, 40.0)]
    for idx, (x, y) in enumerate(locations):
        row = base.copy()
        row["event_id"] = f"box-{idx}"
        row["action_x"] = x
        row["action_y"] = y
        row["player_id"] = 9000
        row["player"] = "Box Tester"
        row["team"] = "Team B"
        rows.append(row)
    summary = build_player_summary(pd.DataFrame(rows), min_actions=1)
    assert int(summary.loc[0, "box_defence_actions"]) == 1
    assert int(summary.loc[0, "box_defence_denominator"]) == 3
    assert summary.loc[0, "box_defence_share"] == 1 / 3


def test_cli_chart_orchestration_creates_expected_files(tmp_path: Path) -> None:
    import subprocess
    import sys

    actions_path = tmp_path / "player.parquet"
    summary_path = tmp_path / "summary.parquet"
    matrix_path = tmp_path / "matrix.parquet"
    config_path = tmp_path / "analysis.yaml"
    production_player_actions_fixture(players=12, actions_per_player=4).to_parquet(actions_path, index=False)
    config_path.write_text(
        "minimum_player_actions: 2\n"
        "minimum_category_sample_size: 2\n"
        "pitch_grid_dimensions: [6, 4]\n"
        "feature_bins: 5\n"
        "cluster_count_candidates: [2, 3]\n"
        "random_seed: 42\n"
        "scaling_method: standard\n"
        "missing_value_threshold: 0.8\n"
        "correlation_threshold: 0.95\n"
        "output_formats: [csv, parquet, png]\n"
        "chart_dpi: 80\n"
        "enable_umap: false\n"
        "enable_hdbscan: false\n"
        "minimum_action_threshold_sensitivity: [2, 3]\n"
        "clustering_feature_groups:\n"
        "  action_mix: [\"action_family_*_share\"]\n"
        "  phase_mix: [\"phase_*_share\"]\n"
        "  spatial_style: [\"mean_action_x\", \"mean_action_y\", \"median_action_x\", \"median_action_y\", \"action_width_std\", \"zone_*_share\"]\n"
        "  possession_style: [\"possession_win_rate\", \"opponent_possession_end_rate\", \"retained_control_rate\", \"actions_under_opponent_possession_rate\"]\n"
        "  360_context: [\"mean_visible_attacker_count\", \"mean_visible_defender_count\", \"mean_nearest_attacker_distance\", \"mean_nearest_defender_distance\", \"mean_defenders_between_ball_and_attacking_goal\", \"role_known_share\", \"reliable_visibility_share\"]\n"
        "  difficulty_exposure: [\"mean_local_numerical_balance_5m\", \"mean_local_numerical_balance_10m\", \"numerical_disadvantage_5m_share\", \"numerical_disadvantage_10m_share\", \"visibility_limited_share\", \"box_defence_share\"]\n"
        "primary_clustering_feature_groups: [action_mix, phase_mix, spatial_style, possession_style, 360_context, difficulty_exposure]\n"
        "feature_group_sensitivity_sets:\n"
        "  all_configured_groups: [action_mix, phase_mix, spatial_style, possession_style, 360_context, difficulty_exposure]\n"
        "  without_360_context: [action_mix, phase_mix, spatial_style, possession_style, difficulty_exposure]\n"
        "  without_possession_style: [action_mix, phase_mix, spatial_style, 360_context, difficulty_exposure]\n"
        "  action_phase_spatial: [action_mix, phase_mix, spatial_style]\n"
        "  spatial_difficulty: [spatial_style, difficulty_exposure]\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "scripts/analyze_features.py", "--input", str(actions_path), "--output-dir", str(tmp_path / "features"), "--config", str(config_path)], check=True)
    subprocess.run([sys.executable, "scripts/build_player_summary.py", "--input", str(actions_path), "--output", str(summary_path), "--config", str(config_path), "--charts-dir", str(tmp_path / "players")], check=True)
    subprocess.run([sys.executable, "scripts/run_player_clustering.py", "--input", str(summary_path), "--output-dir", str(tmp_path / "clustering"), "--matrix-output", str(matrix_path), "--config", str(config_path), "--actions-input", str(actions_path)], check=True)

    expected = [
        tmp_path / "features" / "event_type_distribution.csv",
        tmp_path / "features" / "player_event_type_distribution.png",
        tmp_path / "spatial" / "all_actions_density.png",
        tmp_path / "spatial" / "all_actions_scatter.png",
        tmp_path / "spatial" / "pressure_density.png",
        tmp_path / "spatial" / "phase_high_press_density.png",
        tmp_path / "players" / "player_action_family_profile.png",
        tmp_path / "players" / "player_phase_profile.png",
        tmp_path / "players" / "activity_vs_outcome_scatter.png",
        tmp_path / "players" / "difficulty_vs_outcome_scatter.png",
        tmp_path / "players" / "possession_win_rate_min_sample.png",
        tmp_path / "players" / "future_xg_vs_action_volume.png",
        tmp_path / "players" / "visibility_reliability_chart.png",
        tmp_path / "clustering" / "cluster_size_chart.png",
        tmp_path / "clustering" / "cluster_centroid_heatmap.png",
        tmp_path / "clustering" / "pca_cluster_scatter.png",
        tmp_path / "clustering" / "pca_loading_chart.png",
        tmp_path / "clustering" / "cluster_stability_chart.png",
        tmp_path / "clustering" / "selected_player_cluster_comparison.png",
        tmp_path / "clustering" / "cluster_threshold_sensitivity.csv",
        tmp_path / "clustering" / "cluster_feature_group_sensitivity.csv",
    ]
    missing = [path for path in expected if not path.exists()]
    assert missing == []
    assert not (tmp_path / "features" / "total_action_pitch_heatmap.png").exists()
    assert not list((tmp_path / "features").glob("action_family_pitch_map_*.png"))
    assert not list((tmp_path / "features").glob("phase_pitch_map_*.png"))


def test_generic_bar_chart_does_not_default_to_45_degree_labels(tmp_path: Path) -> None:
    from dax.analysis.plotting import bar_chart

    fig = bar_chart(pd.DataFrame({"category": ["A", "B"], "value": [1, 2]}), "category", "value", tmp_path / "bars.png", "Bars")
    rotations = {tick.get_rotation() for tick in fig.axes[0].get_xticklabels()}
    assert 45.0 not in rotations


def test_display_labels_and_reduced_spatial_features() -> None:
    from dax.analysis.config import load_analysis_config
    from dax.analysis.plot_style import display_label

    assert display_label("box_defence_share") == "Defensive-box exposure"
    summary = build_player_summary(production_player_actions_fixture(players=8, actions_per_player=3), min_actions=1)
    config = load_analysis_config(None)
    config["minimum_player_actions"] = 1
    _, _, metadata = prepare_clustering_matrix(summary, config)
    assert "central_action_share" in metadata["selected_features"]
    assert not any(feature.startswith("zone_") for feature in metadata["selected_features"])


def test_mplsoccer_pitch_plotting_outputs_and_preserves_input(tmp_path: Path) -> None:
    from dax.analysis.pitch_plotting import plot_pitch_density, plot_pitch_rate_map, plot_pitch_scatter

    df = production_player_actions_fixture(players=4, actions_per_player=3)
    before = df.copy(deep=True)
    plot_pitch_scatter(df, tmp_path / "scatter.png", title="Scatter")
    plot_pitch_density(df, tmp_path / "density.png", title="Density", bins=(12, 8))
    plot_pitch_rate_map(df, tmp_path / "rate.png", value_col="target_future_shot_10s", title="Rate", min_bin_actions=50)
    assert (tmp_path / "scatter.png").exists()
    assert (tmp_path / "density.png").exists()
    assert (tmp_path / "rate.png").exists()
    pd.testing.assert_frame_equal(df, before)


def test_k2_k3_and_position_outputs_created_by_cli(tmp_path: Path) -> None:
    # The broader CLI chart test already runs clustering; assert the named analytical outputs too.
    import subprocess
    import sys

    actions_path = tmp_path / "player.parquet"
    summary_path = tmp_path / "summary.parquet"
    matrix_path = tmp_path / "matrix.parquet"
    production_player_actions_fixture(players=12, actions_per_player=4).to_parquet(actions_path, index=False)
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(Path("configs/analysis.yaml").read_text(encoding="utf-8").replace("minimum_player_actions: 30", "minimum_player_actions: 2").replace("minimum_action_threshold_sensitivity: [20, 30, 40, 50]", "minimum_action_threshold_sensitivity: [2, 3]"), encoding="utf-8")
    subprocess.run([sys.executable, "scripts/build_player_summary.py", "--input", str(actions_path), "--output", str(summary_path), "--config", str(config_path), "--charts-dir", str(tmp_path / "players")], check=True)
    subprocess.run([sys.executable, "scripts/run_player_clustering.py", "--input", str(summary_path), "--output-dir", str(tmp_path / "clustering"), "--matrix-output", str(matrix_path), "--config", str(config_path), "--actions-input", str(actions_path)], check=True)
    assert (tmp_path / "clustering" / "k2_k3_comparison.csv").exists()
    assert (tmp_path / "clustering" / "k2_k3_interpretation.md").exists()
    assert (tmp_path / "clustering" / "by_position").exists()
    assert (tmp_path / "clustering" / "cluster_0_spatial_profile.png").exists()


def test_representative_players_use_centroid_distance() -> None:
    from dax.analysis.clustering import representative_players_from_centroids

    matrix = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "team": ["A", "A", "A"],
            "player_name": ["near", "far", "other"],
            "total_actions": [10, 100, 10],
            "matches": [1, 1, 1],
            "feature_a": [0.0, 10.0, 100.0],
            "feature_b": [0.0, 10.0, 100.0],
        }
    )
    assignments = pd.DataFrame({"player_id": [1, 2, 3], "team": ["A", "A", "A"], "cluster": [0, 0, 1]})
    reps = representative_players_from_centroids(matrix, assignments)
    assert reps.loc[reps["cluster"] == 0, "player_id"].iloc[0] == 1
    assert "centroid_distance" in reps.columns
    assert reps["representative_selection_method"].eq("nearest_centroid").all()


def test_weak_cluster_warnings_are_reported(tmp_path: Path) -> None:
    from dax.analysis.reporting import build_model_readiness

    clustering = tmp_path / "clustering"
    clustering.mkdir()
    pd.DataFrame(
        [
            {"silhouette": 0.1, "size_balance": 0.1, "subsample_ari_stability": 0.95},
        ]
    ).to_csv(clustering / "cluster_evaluation.csv", index=False)
    pd.DataFrame({"explained_variance_ratio": [0.1, 0.1]}).to_csv(clustering / "pca_explained_variance.csv", index=False)
    readiness = build_model_readiness(tmp_path)
    warnings = readiness["clustering_interpretation_warnings"]["warnings"]
    assert any("silhouette" in warning for warning in warnings)
    assert any("stability does not imply" in warning for warning in warnings)


def test_bar_chart_uses_cluster_colour_mapping(tmp_path: Path) -> None:
    from matplotlib.colors import to_hex

    from dax.analysis.plot_style import CLUSTER_COLOURS

    fig = bar_chart(
        pd.DataFrame({"cluster": [0, 1], "players": [5, 7]}),
        "cluster",
        "players",
        tmp_path / "cluster_bars.png",
        "Cluster bars",
        color="cluster",
        force_vertical=True,
    )
    colours = [to_hex(patch.get_facecolor()) for patch in fig.axes[0].patches]
    assert colours[:2] == [CLUSTER_COLOURS[0], CLUSTER_COLOURS[1]]


def test_position_aware_clustering_eligibility_uses_action_threshold(tmp_path: Path) -> None:
    from scripts.run_player_clustering import _run_position_aware_clustering

    summary = build_player_summary(production_player_actions_fixture(players=12, actions_per_player=4), min_actions=1)
    summary["position_group"] = ["centre_back"] * 6 + ["midfielder"] * 4 + ["forward"] * 2
    summary.loc[summary.index[:2], "total_actions"] = 1
    config = load_analysis_config(None)
    config["minimum_player_actions"] = 4
    config["cluster_count_candidates"] = [2, 3]
    config["chart_dpi"] = 80

    _run_position_aware_clustering(summary, config, tmp_path)
    defenders = pd.read_csv(tmp_path / "by_position" / "defenders" / "cluster_evaluation.csv")
    assert {"total_players", "eligible_players", "excluded_players", "minimum_player_actions"}.issubset(defenders.columns)
    assert defenders["total_players"].iloc[0] == 6
    assert defenders["eligible_players"].iloc[0] == 4
    assert defenders["excluded_players"].iloc[0] == 2
    assert defenders["status"].iloc[0] == "skipped"
