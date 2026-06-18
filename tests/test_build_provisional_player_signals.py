from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_provisional_player_signals.py"
SPEC = importlib.util.spec_from_file_location("build_provisional_player_signals_script", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_player_aggregates_uses_two_part_oof_schema():
    two_part_oof = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "match_id": [1, 1, 2],
            "player_id": [10, 10, 20],
            "team": ["A", "A", "B"],
            "action_family": ["pressure", "recovery", "pressure"],
            "phase_label": ["phase_1", "phase_1", "phase_2"],
            "position_group": ["midfielder", "midfielder", "centre_back"],
            "oof_shot_probability": [0.2, 0.1, 0.4],
            "observed_future_shot": [1, 0, 0],
            "observed_future_xg": [0.3, 0.0, 0.0],
            "conditional_xg_prediction": [0.5, 0.2, 0.1],
            "combined_future_xg_prediction": [0.10, 0.02, 0.04],
        }
    )

    player_signals = MODULE.build_player_aggregates(two_part_oof)
    player_a = player_signals.loc[player_signals["player_id"].eq(10)].iloc[0]
    player_b = player_signals.loc[player_signals["player_id"].eq(20)].iloc[0]

    assert player_a["eligible_actions"] == 2
    assert player_a["represented_matches"] == 1
    assert player_a["expected_shots"] == pytest.approx(0.3)
    assert player_a["expected_future_xg"] == pytest.approx(0.12)
    assert player_a["observed_future_xg"] == pytest.approx(0.3)
    assert player_a["actions_pressure"] == 1
    assert player_a["xg_pressure"] == pytest.approx(0.10)
    assert player_a["actions_recovery"] == 1
    assert player_a["actions_phase_1"] == 2
    assert player_a["actions_midfielder"] == 2
    assert player_a["conditional_severity_suppression"] == pytest.approx(0.2)

    assert player_b["eligible_actions"] == 1
    assert player_b["expected_future_xg"] == pytest.approx(0.04)
    assert player_b["actions_pressure"] == 1
    assert player_b["xg_pressure"] == pytest.approx(0.04)
    assert player_b["conditional_severity_suppression"] == 0


def test_compute_sensitivity_comparisons_merges_regression_totals_by_player():
    two_part_oof = pd.DataFrame(
        {
            "player_id": [10, 10, 20],
            "team": ["A", "A", "B"],
            "combined_future_xg_prediction": [0.1, 0.2, 0.3],
            "observed_future_xg": [0.0, 0.1, 0.2],
        }
    )
    regression_oof = pd.DataFrame(
        {
            "player_id": [10, 10, 20],
            "team": ["A", "A", "B"],
            "y_pred": [0.2, 0.2, 0.5],
            "y_true": [0.0, 0.1, 0.2],
        }
    )

    sensitivity = MODULE.compute_sensitivity_comparisons(two_part_oof, regression_oof)
    player_a = sensitivity.loc[sensitivity["player_id"].eq(10)].iloc[0]

    assert player_a["expected_xg_two_part"] == pytest.approx(0.3)
    assert player_a["suppression_two_part"] == pytest.approx(0.2)
    assert player_a["expected_xg_regression"] == pytest.approx(0.4)
    assert player_a["suppression_regression"] == pytest.approx(0.3)
    assert "spearman_suppression_correlation" in sensitivity.columns


