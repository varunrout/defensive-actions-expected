from pathlib import Path
import importlib.util

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from dax.coach_analysis.bootstrap import match_level_bootstrap_ci
from dax.coach_analysis.loaders import CoachAnalysisInputError, oof_coverage, select_required_two_part, select_required_variant, validate_unique_predictions, validate_schema
from dax.coach_analysis.plotting import horizontal_metric_chart
from dax.coach_analysis.populations import box_defence_population
from dax.coach_analysis.reporting import data_derived_conclusions, write_markdown_report
from dax.coach_analysis.timeline import add_next_events, validate_processed_timeline
from dax.coach_analysis.visibility import add_reliable_visibility, visibility_report
from dax.coach_analysis.zones import add_pitch_zones, box_zone


def _load_script(name: str):
    path = Path("scripts/coach_analysis") / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_actions():
    return pd.DataFrame({"match_id":[1,1,2,2],"event_id":[10,11,20,21],"x":[115,111,106,104],"y":[40,42,40,20],"position_group":["Centre Back","CB","Full Back","Centre Back"],"competition_label":["World Cup","World Cup","Euros","Euros"],"action_family":["Clearance","Block","Pressure","Duel"],"event_type":["Clearance","Block","Pressure","Duel"],"possession":[5,5,6,6],"team":["A","A","B","B"],"observed_future_xg":[0.0,.2,.05,.1],"expected_future_xg":[.1,.05,.03,.08],"observed_future_shot":[0,1,0,1],"expected_shot_probability":[.2,.1,.1,.2],"has_360":[True, False, True, True],"local_5m_region_fully_visible":[True, True, True, False],"local_10m_region_fully_visible":[True, True, True, True],"freeze_frame_roles_known":[True, True, False, True]})


def sample_events():
    return pd.DataFrame({"match_id":[1,1,1,2,2,2],"event_id":[10,11,12,20,21,22],"period":[1,1,1,1,1,1],"minute":[1,1,1,2,2,2],"second":[1,2,3,1,2,3],"team":["A","B","B","B","B","C"],"possession":[5,5,5,6,6,7],"event_type":["Clearance","Block","Shot","Pressure","Duel","Pass"],"x":[115,111,110,106,115,104],"y":[40,42,40,40,20,40]})


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


def test_oof_coverage_and_sensitivities():
    preds = pd.DataFrame({"match_id":[1,2],"event_id":[10,20],"fold":[0,1]})
    coverage = oof_coverage(sample_actions(), preds)
    assert coverage["missing_predictions"] == 2
    assert coverage["folds"] == [0, 1]


def test_boolean_visibility_coverage():
    out = add_reliable_visibility(sample_actions())
    assert out["coach_reliable_visibility"].sum() == 1
    report = visibility_report(sample_actions())
    assert report["columns"]["has_360"]["true"] == 3


def test_processed_event_timeline_validation():
    report = validate_processed_timeline(sample_events())
    assert report["next_event_sequence_possible"] is True
    assert report["duplicate_event_ids"] == 0


def test_true_sequence_outcomes_for_clearance_block_pressure():
    cb = _load_script("01_analyze_cb_box_defence.py")
    joined = add_next_events(sample_actions(), sample_events())
    pop = cb._augment_sequences(box_defence_population(joined, centre_backs_only=True))
    assert "clearance recycled" in set(pop["coach_clearance_outcome"])
    assert "block followed by rebound" in set(pop["coach_block_outcome"])
    pressure = cb._augment_sequences(joined[joined.event_type.eq("Pressure")])
    assert "pressure followed by progression" in set(pressure["coach_pressure_outcome"])


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
    assert box_zone(115, 40) == "six-yard box"
    assert {"six-yard box", "penalty-spot zone", "wide box"}.issubset(set(zones["coach_box_zone"]))


def test_report_generation_and_no_hardcoded_football_conclusions(tmp_path):
    path = tmp_path / "report.md"
    write_markdown_report(path, "Title", [("Section", "Content")])
    assert "# Title" in path.read_text()
    table = pd.DataFrame({"subgroup":["A","B"],"metric":[.2,.1],"actions":[40,10],"matches":[6,2],"ci_low":[.1,0],"ci_high":[.3,.2]})
    conclusions = data_derived_conclusions(table, "subgroup", "metric")
    assert conclusions[0]["subgroup"] == "A"
    assert all("subgroup" in c and "metric" in c for c in conclusions)
