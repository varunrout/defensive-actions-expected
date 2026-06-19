from pathlib import Path
import importlib.util

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from dax.coach_analysis.bootstrap import match_level_bootstrap_ci
from dax.coach_analysis.loaders import oof_coverage, select_variant, validate_oof_alignment, validate_schema
from dax.coach_analysis.plotting import horizontal_metric_chart
from dax.coach_analysis.populations import box_defence_population
from dax.coach_analysis.reporting import data_derived_conclusions, write_markdown_report
from dax.coach_analysis.zones import add_pitch_zones, box_zone


def _load_script(name: str):
    path = Path("scripts/coach_analysis") / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_actions():
    return pd.DataFrame(
        {
            "match_id": [1, 1, 2, 2, 3],
            "event_id": [10, 11, 20, 21, 30],
            "x": [115, 111, 106, 104, 90],
            "y": [40, 42, 40, 20, 10],
            "position_group": ["Centre Back", "CB", "Full Back", "Centre Back", "CB"],
            "competition": ["World Cup", "World Cup", "Euros", "Euros", "Euros"],
            "action_family": ["Clearance", "Block", "Pressure", "Duel", "Pressure"],
            "event_type": ["Clearance", "Block", "Pressure", "Duel", "Pressure"],
            "possession_won": [True, False, False, False, False],
            "observed_future_xg": [0.00, 0.20, 0.05, 0.10, 0.01],
            "expected_future_xg": [0.10, 0.05, 0.03, 0.08, 0.02],
            "observed_future_shot": [0, 1, 0, 1, 0],
            "expected_shot_probability": [0.2, 0.1, 0.1, 0.2, 0.1],
            "visible_teammates": [3, 4, None, 2, None],
        }
    )


def test_cli_parsing_defaults():
    script = _load_script("00_check_coach_analysis_readiness.py")
    args = script.parse_args([])
    assert args.classification_variant == "b7_full_with_360"
    assert args.regression_variant == "r4_full_with_360"


def test_schema_validation():
    result = validate_schema(sample_actions(), [["match_id", "event_id"]], ["competition"])
    assert result["valid"] is True
    assert result["missing_recommended"] == []


def test_explicit_model_selection():
    df = pd.DataFrame({"variant": ["b7_full_with_360", "other"], "event_id": [1, 2]})
    selected = select_variant(df, "b7_full_with_360")
    assert selected["event_id"].tolist() == [1]


def test_duplicate_oof_rejection_signal():
    actions = sample_actions()
    preds = pd.DataFrame({"match_id": [1, 1], "event_id": [10, 10], "fold": [0, 0]})
    assert validate_oof_alignment(actions, preds)["duplicate_predictions"] == 1


def test_oof_coverage_missing_predictions_and_folds():
    actions = sample_actions()
    preds = pd.DataFrame({"match_id": [1, 2], "event_id": [10, 20], "fold": [0, 1]})
    coverage = oof_coverage(actions, preds)
    assert coverage["missing_predictions"] == 3
    assert coverage["folds"] == [0, 1]


def test_match_level_bootstrap():
    result = match_level_bootstrap_ci(sample_actions(), "observed_future_xg", n_boot=50, seed=1)
    assert result["matches"] == 3
    assert result["actions"] == 5
    assert result["ci_low"] <= result["value"] <= result["ci_high"]


def test_report_generation(tmp_path):
    path = tmp_path / "report.md"
    write_markdown_report(path, "Title", [("Section", "Content")])
    assert "# Title" in path.read_text()


def test_horizontal_plotting(tmp_path):
    path = tmp_path / "plot.png"
    _, ax = horizontal_metric_chart(pd.DataFrame({"label": ["A", "B"], "value": [1, 2]}), "label", "value", "Title", path)
    assert path.exists()
    assert all(t.get_rotation() == 0 for t in ax.get_xticklabels())


def test_empty_populations():
    assert box_defence_population(sample_actions().iloc[0:0], centre_backs_only=True).empty


def test_cb_filtering():
    pop = box_defence_population(sample_actions(), centre_backs_only=True)
    assert set(pop["position_group"]) == {"Centre Back", "CB"}


def test_box_zone_analysis():
    zones = add_pitch_zones(sample_actions())
    assert box_zone(115, 40) == "six-yard box"
    assert {"six-yard box", "penalty-spot zone", "wide box"}.issubset(set(zones["coach_box_zone"]))


def test_data_derived_conclusions():
    table = pd.DataFrame({"subgroup": ["A", "B"], "metric": [0.2, 0.1], "actions": [40, 10], "matches": [6, 2], "ci_low": [0.1, 0.0], "ci_high": [0.3, 0.2]})
    conclusions = data_derived_conclusions(table, "subgroup", "metric")
    assert conclusions[0]["subgroup"] == "A"
    assert conclusions[0]["difference"] is not None
