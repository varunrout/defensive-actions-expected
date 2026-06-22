from pathlib import Path
import importlib.util

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from dax.coach_analysis.bootstrap import match_level_bootstrap_ci
from dax.coach_analysis.loaders import CoachAnalysisInputError, add_canonical_model_columns, input_status_from_paths, oof_coverage_for_variant, select_required_two_part, select_required_variant, validate_unique_predictions, validate_schema
from dax.coach_analysis.plotting import horizontal_metric_chart
from dax.coach_analysis.populations import box_defence_population
from dax.coach_analysis.reporting import data_derived_conclusions, write_markdown_report
from dax.coach_analysis.timeline import add_next_events, normalise_processed_events, validate_processed_timeline
from dax.coach_analysis.visibility import add_reliable_visibility, visibility_report
from dax.coach_analysis.zones import CoordinateError, add_pitch_zones, attacking_box_zone, defensive_box_zone, find_xy_columns


def _load_script(name: str):
    path = Path("scripts/coach_analysis") / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_actions():
    return pd.DataFrame({"match_id":[1,1,2,2],"event_id":[10,11,20,21],"action_x":[5,11,106,17],"action_y":[40,42,40,20],"position_group":["Centre Back","CB","Full Back","Centre Back"],"competition_label":["World Cup","World Cup","Euros","Euros"],"action_family":["Clearance","Block","Pressure","Duel"],"event_type":["Clearance","Block","Pressure","Duel"],"possession":[5,5,6,6],"team":["A","A","B","B"],"observed_future_xg":[0.0,.2,.05,.1],"expected_future_xg":[.1,.05,.03,.08],"observed_future_shot":[0,1,0,1],"expected_shot_probability":[.2,.1,.1,.2],"has_360":[True, False, True, True],"local_5m_region_fully_visible":[True, True, True, False],"local_10m_region_fully_visible":[True, True, True, True],"freeze_frame_roles_known":[True, True, False, True]})


def sample_events():
    return pd.DataFrame({"match_id":[1,1,1,2,2,2],"event_id":[10,11,12,20,21,22],"period":[1,1,1,1,1,1],"minute":[1,1,1,2,2,2],"second":[1,2,3,1,2,3],"team":["A","B","B","B","C","C"],"possession":[5,5,5,6,6,7],"event_type":["Clearance","Block","Shot","Pressure","Duel","Pass"],"x":[115,111,110,106,115,104],"y":[40,42,40,40,20,40]})


def test_explicit_path_parsing_and_defaults():
    script = _load_script("00_check_coach_analysis_readiness.py")
    args = script.parse_args(["--actions-input", "custom.parquet", "--processed-events-input", "events.parquet"])
    assert str(args.actions_input) == "custom.parquet"
    assert args.two_part_classification_variant == "b7_full_with_360"
    assert args.two_part_conditional_variant == "conditional_tweedie"


def test_missing_required_inputs_return_nonzero_and_allow_partial(tmp_path):
    script = _load_script("00_check_coach_analysis_readiness.py")
    out = tmp_path / "out"
    code = script.main(["--repo-root", str(tmp_path), "--output-root", str(out)])
    assert code != 0
    code_partial = script.main(["--repo-root", str(tmp_path), "--output-root", str(out), "--allow-partial"])
    assert code_partial == 0


def test_schema_validation():
    assert validate_schema(sample_actions(), [["match_id", "event_id"]], ["competition_label"])["valid"] is True


def test_strict_unavailable_variant_failure_and_selection():
    df = pd.DataFrame({"model_variant":["b7_full_with_360"],"match_id":[1],"event_id":[1]})
    assert len(select_required_variant(df, "b7_full_with_360")) == 1
    try:
        select_required_variant(df, "b6_full_without_360")
    except CoachAnalysisInputError as exc:
        assert "Available variants" in str(exc)
    else:
        raise AssertionError("expected unavailable variant failure")


def test_explicit_two_part_pair_selection():
    df = pd.DataFrame({"classification_model_variant":["b7_full_with_360","b7_full_with_360"],"conditional_model_variant":["conditional_tweedie","other"],"match_id":[1,2],"event_id":[1,2]})
    assert select_required_two_part(df, "b7_full_with_360", "conditional_tweedie")["event_id"].tolist() == [1]


def test_duplicate_prediction_rejection():
    preds = pd.DataFrame({"match_id":[1,1],"event_id":[10,10]})
    try:
        validate_unique_predictions(preds)
    except CoachAnalysisInputError as exc:
        assert "duplicate predictions" in str(exc)
    else:
        raise AssertionError("expected duplicate failure")


def test_360_native_population_coverage_and_custom_input_status(tmp_path):
    preds = pd.DataFrame({"match_id":[1,2],"event_id":[10,20],"fold":[0,1]})
    coverage = oof_coverage_for_variant(sample_actions(), preds, "b7_full_with_360")
    assert coverage["native_eligible_actions"] == 2
    assert coverage["missing_eligible_predictions"] == 0
    assert "derived from selected OOF event IDs" in coverage["eligibility_method"]
    status = input_status_from_paths({"actions": Path("custom.parquet")}, tmp_path)
    assert status.loc[0, "path"] == "custom.parquet"


def test_boolean_visibility_coverage():
    out = add_reliable_visibility(sample_actions())
    assert out["coach_reliable_visibility"].sum() == 1
    report = visibility_report(sample_actions())
    assert report["columns"]["has_360"]["true"] == 3


def test_processed_event_timeline_validation_id_normalisation_and_no_cross_period():
    events = sample_events().rename(columns={"event_id": "id"})
    normalised = normalise_processed_events(events)
    assert "event_id" in normalised.columns
    report = validate_processed_timeline(events)
    assert report["next_event_sequence_possible"] is True
    assert report["duplicate_event_ids"] == 0
    two_periods = pd.DataFrame({"match_id":[1,1],"event_id":[1,2],"period":[1,2],"minute":[45,45],"second":[0,1]})
    linked = add_next_events(two_periods, two_periods)
    assert linked.loc[linked.event_id.eq(1), "next_event_id"].isna().all()


def test_true_sequence_outcomes_for_clearance_block_pressure():
    cb = _load_script("01_analyze_cb_box_defence.py")
    joined = add_next_events(sample_actions(), sample_events())
    pop = cb._augment_sequences(box_defence_population(joined, centre_backs_only=True))
    assert "opposition recycle" in set(pop["coach_clearance_outcome"])
    assert "block followed by opposition rebound or shot" in set(pop["coach_block_outcome"])
    pressure = cb._augment_sequences(pd.DataFrame({"event_type": ["Pressure"], "team": ["A"], "next_team": ["B"], "action_x": [10], "x": [10], "next_x": [20], "next_event_within_window": [True]}))
    assert "opposition progresses" in set(pressure["coach_pressure_outcome"])


def test_competition_label_handling():
    cb = _load_script("01_analyze_cb_box_defence.py")
    out = cb._canonical_competition(sample_actions())
    assert set(out["coach_competition"]) == {"World Cup", "Euros"}


def test_cli_sample_thresholds_in_conclusions():
    table = pd.DataFrame({"subgroup":["A"],"metric":[.2],"actions":[5],"matches":[1],"ci_low":[.1],"ci_high":[.3]})
    conclusions = data_derived_conclusions(table, "subgroup", "metric", min_actions=10, min_matches=2)
    assert "5 < 10" in conclusions[0]["minimum_sample_warning"]


def test_plotting_bootstrap_empty_cb_box_and_zones(tmp_path):
    assert match_level_bootstrap_ci(sample_actions(), "observed_future_xg", n_boot=20)["matches"] == 2
    path = tmp_path / "plot.png"
    _, ax = horizontal_metric_chart(pd.DataFrame({"label":["A"],"value":[1]}), "label", "value", "Title", path)
    assert path.exists() and all(t.get_rotation() == 0 for t in ax.get_xticklabels())
    assert box_defence_population(sample_actions().iloc[0:0], centre_backs_only=True).empty
    assert set(box_defence_population(sample_actions(), centre_backs_only=True)["position_group"]) == {"Centre Back", "CB"}
    zones = add_pitch_zones(sample_actions())
    assert defensive_box_zone(5, 40) == "own six-yard box"
    assert {"own six-yard box", "own penalty-spot/deep central zone", "own left/wide box"}.issubset(set(zones["coach_defensive_box_zone"]))


def test_report_generation_and_no_hardcoded_football_conclusions(tmp_path):
    path = tmp_path / "report.md"
    write_markdown_report(path, "Title", [("Section", "Content")])
    assert "# Title" in path.read_text()
    table = pd.DataFrame({"subgroup":["A","B"],"metric":[.2,.1],"actions":[40,10],"matches":[6,2],"ci_low":[.1,0],"ci_high":[.3,.2]})
    conclusions = data_derived_conclusions(table, "subgroup", "metric")
    assert conclusions[0]["subgroup"] == "A"
    assert all("subgroup" in c and "metric" in c for c in conclusions)


def test_actual_oof_column_mapping_and_canonical_suppression():
    frame = pd.DataFrame({"b7_y_score":[.3],"b6_y_score":[.2],"r4_y_pred":[.05],"r6_y_pred":[.04],"two_part_combined_future_xg_prediction":[.06],"two_part_observed_future_shot":[0],"two_part_observed_future_xg":[0.0]})
    out = add_canonical_model_columns(frame)
    assert out.loc[0, "coach_expected_shot_b7"] == .3
    assert out.loc[0, "coach_expected_xg_two_part"] == .06
    assert out.loc[0, "coach_observed_xg"] == 0.0


def test_clearance_out_of_play_not_recycled_and_repeated_window_outputs(tmp_path):
    cb = _load_script("01_analyze_cb_box_defence.py")
    actions = sample_actions().iloc[[0]].copy()
    events = pd.DataFrame({"match_id":[1,1],"event_id":[10,99],"period":[1,1],"minute":[1,1],"second":[1,2],"team":["A","B"],"possession":[5,5],"event_type":["Clearance","Out"],"x":[5,120],"y":[40,80]})
    pop = cb._augment_sequences(add_next_events(actions, events))
    assert pop.loc[0, "coach_clearance_outcome"] == "out of play"
    counts = cb._write_category_video_files(pd.DataFrame(), tmp_path)
    assert "clearance_relief.csv" in counts
    assert (tmp_path / "block_rebound.csv").exists()


def test_separated_conclusion_semantics():
    table = pd.DataFrame({"subgroup":["danger"],"future_xg_per_action":[.2],"actions":[50],"matches":[5],"ci_low":[.1],"ci_high":[.3]})
    conclusions = data_derived_conclusions(table, "subgroup", "future_xg_per_action")
    assert conclusions[0]["metric"] == "future_xg_per_action"


def test_grouped_bootstrap_ci_non_null_for_canonical_observed_xg():
    from dax.coach_analysis.bootstrap import add_match_bootstrap_by_group

    frame = pd.DataFrame({
        "match_id": [1, 1, 2, 2, 3, 3],
        "group": ["A", "A", "A", "A", "A", "A"],
        "coach_observed_xg": [0.0, 0.1, 0.2, 0.0, 0.1, 0.3],
    })
    out = add_match_bootstrap_by_group(frame, ["group"], "coach_observed_xg", n_boot=50, seed=2)
    assert out.loc[0, "ci_low"] == out.loc[0, "ci_low"]
    assert out.loc[0, "ci_high"] == out.loc[0, "ci_high"]


def test_possession_control_canonical_from_real_feature_names():
    frame = pd.DataFrame({
        "action_won_possession": [False, True, False],
        "action_changed_possession": [False, False, True],
        "possession_won": [False, False, False],
        "won_possession": [False, False, False],
    })
    out = add_canonical_model_columns(frame)
    assert out["coach_possession_controlled"].tolist() == [False, True, True]


def test_default_processed_event_path_is_pipeline_output():
    script = _load_script("00_check_coach_analysis_readiness.py")
    args = script.parse_args([])
    assert args.processed_events_input == Path("data/processed/events_with_targets.parquet")


def test_pressure_and_block_respect_time_window():
    cb = _load_script("01_analyze_cb_box_defence.py")
    actions = pd.DataFrame({
        "match_id": [1, 1], "event_id": [1, 3], "period": [1, 1],
        "action_x": [10, 12], "action_y": [40, 40], "team": ["A", "A"],
        "event_type": ["Pressure", "Block"], "position_group": ["CB", "CB"],
        "possession": [1, 1],
    })
    events = pd.DataFrame({
        "match_id": [1, 1, 1, 1], "event_id": [1, 2, 3, 4], "period": [1, 1, 1, 1],
        "minute": [0, 0, 0, 0], "second": [0, 20, 30, 45],
        "team": ["A", "B", "A", "B"], "possession": [1, 1, 1, 1],
        "event_type": ["Pressure", "Pass", "Block", "Shot"], "x": [50, 80, 100, 110], "y": [40, 40, 40, 40],
    })
    pop = cb._augment_sequences(add_next_events(actions, events, window_seconds=10), window_seconds=10)
    assert "no immediate continuation observed" in set(pop["coach_pressure_outcome"])
    assert "unresolved" in set(pop["coach_block_outcome"])


def test_action_xy_discovery_and_defensive_attacking_zone_separation():
    df = pd.DataFrame({"action_x": [5, 110], "action_y": [40, 40]})
    assert find_xy_columns(df, strict=True) == ("action_x", "action_y")
    zones = add_pitch_zones(df, strict=True)
    assert zones.loc[0, "coach_defensive_box_zone"] == "own six-yard box"
    assert zones.loc[1, "coach_defensive_box_zone"] == "outside defensive box"
    assert attacking_box_zone(110, 40).startswith("attacking")
    assert defensive_box_zone(110, 40) == "outside defensive box"


def test_missing_coordinates_fail_clearly():
    try:
        add_pitch_zones(pd.DataFrame({"a": [1]}), strict=True)
    except CoordinateError as exc:
        assert "No supported coordinate pair" in str(exc)
    else:
        raise AssertionError("expected coordinate failure")


def test_spatial_versus_phase_population_overlap_and_empty_cb_nonzero(tmp_path):
    cb = _load_script("01_analyze_cb_box_defence.py")
    actions = add_pitch_zones(pd.DataFrame({"match_id": [1, 1], "event_id": [1, 2], "action_x": [5, 50], "action_y": [40, 40], "position_group": ["centre_back", "centre_back"], "phase_label": ["box_defence", "box_defence"]}), strict=True)
    counts = cb._population_stage_counts(actions, actions.iloc[[0]])
    assert counts["centre_back_own_box_actions"] == 1
    assert counts["centre_back_phase_labelled_box_defence_actions"] == 2
    assert counts["spatial_phase_overlap"] == 1
    out = tmp_path / "out"
    code = cb.main(["--repo-root", str(tmp_path), "--output-root", str(out)])
    assert code != 0


def test_defensive_plot_half_orientation(tmp_path):
    from dax.coach_analysis.plotting import action_pitch_map
    df = pd.DataFrame({"action_x": [5, 10], "action_y": [40, 45], "match_id": [1, 1]})
    _, ax = action_pitch_map(df, "Own-box test", tmp_path / "plot.png")
    assert ax.get_xlim()[0] <= 0 and ax.get_xlim()[1] <= 60


def _write_minimal_cli_inputs(tmp_path: Path, actions: pd.DataFrame) -> dict[str, Path]:
    events = pd.DataFrame({
        "match_id": [1, 1], "event_id": [1, 2], "period": [1, 1], "minute": [0, 0], "second": [1, 2],
        "team": ["A", "B"], "possession": [1, 1], "event_type": ["Clearance", "Pass"], "x": [5, 20], "y": [40, 40],
    })
    cls = pd.DataFrame({
        "match_id": [1, 1], "event_id": [1, 1], "model_variant": ["b7_full_with_360", "b6_full_without_360"],
        "y_score": [0.2, 0.1], "y_true": [0, 0], "fold": [0, 0],
    })
    reg = pd.DataFrame({
        "match_id": [1, 1], "event_id": [1, 1], "model_variant": ["r4_full_with_360", "r6_nonlinear_candidate"],
        "y_pred": [0.05, 0.04], "y_true": [0.0, 0.0], "fold": [0, 0],
    })
    two = pd.DataFrame({
        "match_id": [1], "event_id": [1], "classification_model_variant": ["b7_full_with_360"],
        "conditional_model_variant": ["conditional_tweedie"], "combined_future_xg_prediction": [0.06],
        "observed_future_xg": [0.0], "observed_future_shot": [0], "fold": [0],
    })
    paths = {}
    for name, frame in {"actions": actions, "events": events, "cls": cls, "reg": reg, "two": two}.items():
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    return paths


def test_stage_counts_use_zoned_full_action_frame_and_summary_coordinates(tmp_path):
    cb = _load_script("01_analyze_cb_box_defence.py")
    actions = pd.DataFrame({
        "match_id": [1, 1, 1], "event_id": [1, 3, 4], "action_x": [5, 5, 50], "action_y": [40, 42, 40],
        "position_group": ["centre_back", "full_back", "centre_back"], "phase_label": ["box_defence", "box_defence", "box_defence"],
        "team": ["A", "A", "A"], "possession": [1, 1, 2], "event_type": ["Clearance", "Block", "Pressure"],
    })
    zoned = add_pitch_zones(actions, strict=True)
    population = box_defence_population(zoned, centre_backs_only=True)
    counts = cb._population_stage_counts(zoned, population)
    assert counts["own_box_actions"] == 2
    assert counts["centre_back_own_box_actions"] == len(population) == 1
    assert counts["coordinate_x_column"] == "action_x"
    assert counts["coordinate_y_column"] == "action_y"


def test_missing_coordinates_exit_code_2_structured_report(tmp_path):
    cb = _load_script("01_analyze_cb_box_defence.py")
    actions = pd.DataFrame({
        "match_id": [1], "event_id": [1], "position_group": ["centre_back"], "phase_label": ["box_defence"],
        "team": ["A"], "possession": [1], "event_type": ["Clearance"],
    })
    paths = _write_minimal_cli_inputs(tmp_path, actions)
    out = tmp_path / "out"
    code = cb.main([
        "--repo-root", str(tmp_path), "--actions-input", str(paths["actions"]), "--processed-events-input", str(paths["events"]),
        "--classification-oof", str(paths["cls"]), "--regression-oof", str(paths["reg"]), "--two-part-oof", str(paths["two"]),
        "--output-root", str(out),
    ])
    assert code == 2
    summary = (out / "execution_summary.json").read_text()
    assert "No supported coordinate pair" in summary
    assert (out / "report.md").exists()


def test_immediate_re_turnover_missing_second_next_is_na_safe():
    cb = _load_script("01_analyze_cb_box_defence.py")
    frame = pd.DataFrame({
        "event_type": ["Clearance"],
        "team": ["A"],
        "next_team": ["A"],
        "possession_won": [True],
    })
    out = cb._augment_sequences(frame)
    assert out.loc[0, "coach_immediate_re_turnover"] is False or not bool(out.loc[0, "coach_immediate_re_turnover"])


def test_competition_metadata_propagation_and_unknown_fallback():
    cb = _load_script("01_analyze_cb_box_defence.py")
    frame = pd.DataFrame({
        "competition_label": ["World Cup 2022", None, "", "nan"],
        "competition": ["ignored", "Euro 2024", "Copa", None],
        "competition_name": ["ignored", "ignored", "ignored", "Fallback Cup"],
        "competition_stage": ["Group Stage", None, "", "nan"],
        "stage": ["ignored", "Final", "Semi-final", None],
        "stage_name": ["ignored", "ignored", "ignored", "Quarter-final"],
        "season_name": ["2022", None, "", "nan"],
        "season": ["ignored", "2024", None, "2025"],
        "season_id": [106, 282, 7, 9],
        "competition_id": [43, None, "", "nan"],
    })
    out = cb._canonical_competition(frame)
    assert out["coach_competition"].tolist() == ["World Cup 2022", "Euro 2024", "Copa", "Fallback Cup"]
    assert out["coach_competition_stage"].tolist() == ["Group Stage", "Final", "Semi-final", "Quarter-final"]
    assert out["coach_season"].tolist() == ["2022", "2024", "7", "2025"]
    assert out["coach_competition_id"].tolist() == ["43", "unknown", "unknown", "unknown"]
    assert cb._canonical_competition(pd.DataFrame({"match_id": [1]})).loc[0, "coach_competition"] == "unknown"


def test_phase_spatial_diagnostic_labels_primary_population():
    cb = _load_script("01_analyze_cb_box_defence.py")
    diagnostic = cb._phase_spatial_diagnostic({
        "centre_back_own_box_actions": 1442,
        "centre_back_phase_labelled_box_defence_actions": 2615,
        "spatial_phase_overlap": 0,
        "coordinate_x_column": "action_x",
        "coordinate_y_column": "action_y",
    })
    assert diagnostic["primary_population"] == "spatial_own_box_centre_back_actions"
    assert diagnostic["spatial_own_box_centre_back_actions"] == 1442
    assert diagnostic["phase_labelled_box_defence_centre_back_actions"] == 2615
    assert diagnostic["overlap_count"] == 0
    assert "not forced to match" in diagnostic["interpretation"]


def test_identical_sensitivity_variant_warning_behavior():
    cb = _load_script("01_analyze_cb_box_defence.py")
    population = pd.DataFrame({
        "match_id": [1, 1],
        "coach_expected_shot_b7": [0.2, 0.3],
        "coach_expected_shot_b6": [0.2, 0.3],
        "coach_expected_xg_r4": [0.01, 0.02],
        "coach_expected_xg_r6": [0.01, 0.02],
        "coach_expected_xg_two_part": [0.03, 0.04],
    })
    tables = cb._sensitivity_tables(population, min_actions=1, min_matches=1)
    warning = tables["b7_vs_b6_expected_shot"].loc[0, "warning"]
    assert tables["b7_vs_b6_expected_shot"].loc[0, "mean_difference"] == 0.0
    assert "identical to the primary variant" in warning
    assert "do not interpret zero disagreement as model robustness" in warning


def test_smoke_sized_sensitivity_warns_below_report_thresholds():
    cb = _load_script("01_analyze_cb_box_defence.py")
    population = pd.DataFrame({
        "match_id": [1, 2, 3, 4] * 5,
        "coach_expected_shot_b7": [0.2] * 20,
        "coach_expected_shot_b6": [0.1] * 20,
        "coach_expected_xg_r4": [0.01] * 20,
        "coach_expected_xg_r6": [0.02] * 20,
        "coach_expected_xg_two_part": [0.03] * 20,
    })
    tables = cb._sensitivity_tables(population, min_actions=30, min_matches=5)
    warning = tables["b7_vs_b6_expected_shot"].loc[0, "warning"]
    assert "smoke-sized sensitivity action sample: 20 < 30" in warning
    assert "smoke-sized sensitivity match sample: 4 < 5" in warning
