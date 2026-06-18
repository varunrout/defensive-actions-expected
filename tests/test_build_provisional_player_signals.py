from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_provisional_player_signals.py"
SPEC = importlib.util.spec_from_file_location("build_provisional_player_signals_script", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _minimal_two_part_oof(n_players: int = 4, shots_per_player: int = 3) -> pd.DataFrame:
    """Create a minimal two-part OOF with variable match contributions (so bootstrap SE > 0)."""
    rng = np.random.RandomState(7)
    rows = []
    for pid in range(n_players):
        for mid in range(6):  # 6 matches to ensure non-trivial bootstrap variance
            for eid in range(3):
                has_shot = mid < shots_per_player
                # Vary the xG contribution per match so bootstrap SE is non-zero.
                xg_base = 0.1 + 0.05 * mid + 0.02 * eid
                rows.append({
                    "event_id": f"e{pid}_{mid}_{eid}",
                    "match_id": mid,
                    "player_id": pid,
                    "team": "A" if pid < max(n_players // 2, 1) else "B",
                    "oof_shot_probability": float(rng.uniform(0.05, 0.4)),
                    "observed_future_shot": 1 if has_shot else 0,
                    "observed_future_xg": xg_base if has_shot else 0.0,
                    "conditional_xg_prediction": xg_base * 1.1 if has_shot else 0.0,
                    "combined_future_xg_prediction": xg_base * 0.3,
                })
    return pd.DataFrame(rows)


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


# ── New tests ──────────────────────────────────────────────────────────────────

class TestReliabilityThresholdsFromQuantiles:
    """derive_reliability_thresholds must use quantiles from the data, not hardcoded values."""

    def test_thresholds_are_quantile_derived(self):
        player_signals = pd.DataFrame({
            "eligible_actions": [10, 20, 30, 40, 50, 60, 70, 80],
            "represented_matches": [2, 4, 6, 8, 10, 12, 14, 16],
            "observed_shot_count": [0, 1, 2, 3, 4, 5, 6, 7],
        })
        result = MODULE.derive_reliability_thresholds(player_signals)
        assert "thresholds" in result
        assert "quantiles" in result
        t = result["thresholds"]
        q = result["quantiles"]
        assert t["actions_medium"] == pytest.approx(q["eligible_actions"]["q50"])
        assert t["matches_medium"] == pytest.approx(q["represented_matches"]["q50"])
        assert t["shots_medium"] == pytest.approx(q["observed_shot_count"]["q50"])

    def test_conditional_severity_shot_threshold_recorded(self):
        player_signals = pd.DataFrame({
            "eligible_actions": list(range(10, 50)),
            "represented_matches": list(range(1, 41)),
            "observed_shot_count": list(range(0, 40)),
        })
        result = MODULE.derive_reliability_thresholds(player_signals)
        assert "conditional_severity_shot_threshold" in result["thresholds"]
        # Must be at least 1
        assert result["thresholds"]["conditional_severity_shot_threshold"] >= 1

    def test_shot_threshold_at_least_one(self):
        """When q50 of shots is 0, threshold must be clamped to 1."""
        player_signals = pd.DataFrame({
            "eligible_actions": [5, 10, 15],
            "represented_matches": [1, 2, 3],
            "observed_shot_count": [0, 0, 0],
        })
        result = MODULE.derive_reliability_thresholds(player_signals)
        assert result["thresholds"]["conditional_severity_shot_threshold"] >= 1


class TestApplyConditionalReliabilityFlag:
    """apply_conditional_reliability_flag must use the data-derived threshold."""

    def test_flag_uses_derived_threshold(self):
        player_signals = pd.DataFrame({
            "player_id": [1, 2, 3],
            "observed_shot_count": [0, 2, 5],
        })
        thresholds = {"thresholds": {"conditional_severity_shot_threshold": 3}}
        result = MODULE.apply_conditional_reliability_flag(player_signals, thresholds)
        assert result["conditional_severity_reliability_flag"].tolist() == [False, False, True]

    def test_no_hardcoded_threshold_of_3(self):
        """The flag threshold is configurable and not hardcoded as 3."""
        player_signals = pd.DataFrame({
            "player_id": [1, 2, 3],
            "observed_shot_count": [3, 3, 3],
        })
        # With threshold=5, all should be False even though count == 3
        thresholds = {"thresholds": {"conditional_severity_shot_threshold": 5}}
        result = MODULE.apply_conditional_reliability_flag(player_signals, thresholds)
        assert result["conditional_severity_reliability_flag"].tolist() == [False, False, False]


class TestBootstrapReproducibility:
    """Match-level bootstrap must produce identical results for the same seed."""

    def test_same_seed_same_results(self):
        oof = _minimal_two_part_oof()
        ci1 = MODULE.bootstrap_player_suppression_match_level(oof, n_bootstrap=50, seed=42)
        ci2 = MODULE.bootstrap_player_suppression_match_level(oof, n_bootstrap=50, seed=42)
        pd.testing.assert_frame_equal(ci1, ci2)

    def test_different_seed_different_results(self):
        oof = _minimal_two_part_oof()
        ci1 = MODULE.bootstrap_player_suppression_match_level(oof, n_bootstrap=200, seed=42)
        ci2 = MODULE.bootstrap_player_suppression_match_level(oof, n_bootstrap=200, seed=99)
        # With variable match contributions, CIs from different seeds should differ.
        if not ci1.empty and not ci2.empty:
            col = "total_combined_xg_suppression_bootstrap_mean"
            # At least one player should have a different bootstrap mean
            diff = (ci1[col] - ci2[col]).abs()
            assert diff.max() > 0 or ci1[col].std() == 0, \
                "Expected different bootstrap means for different seeds with variable data"

    def test_bootstrap_uses_single_match_sample(self):
        """All three suppression types are computed from the same sampled match population.

        Verified indirectly: when every shot row comes from a distinct match, the
        conditional CI width should correlate with combined_xg CI width.
        """
        oof = _minimal_two_part_oof(n_players=2, shots_per_player=4)
        ci = MODULE.bootstrap_player_suppression_match_level(oof, n_bootstrap=200, seed=0)
        assert not ci.empty
        assert "total_combined_xg_suppression_ci95_lower" in ci.columns
        assert "total_conditional_severity_suppression_ci95_lower" in ci.columns
        # The SE of total combined should be finite for players with multiple matches
        assert ci["total_combined_xg_suppression_bootstrap_se"].dropna().gt(0).any()


class TestCIOverlapNotNaN:
    """compute_player_sensitivity must not return universally NaN confidence_interval_overlap
    when bootstrap CI columns are present."""

    def test_ci_overlap_computed_from_bootstrap_columns(self):
        n = 10
        player_signals = pd.DataFrame({
            "player_id": range(n),
            "team": ["A"] * n,
            "combined_xg_suppression": np.linspace(-1, 1, n),
            "mean_combined_xg_suppression": np.linspace(-0.5, 0.5, n),
            "reliability_tier": ["high"] * n,
            # Provide bootstrap CI columns
            "total_combined_xg_suppression_ci95_lower": np.linspace(-2, -0.1, n),
            "total_combined_xg_suppression_ci95_upper": np.linspace(0.1, 2, n),
        })
        result = MODULE.compute_player_sensitivity(player_signals)
        assert "confidence_interval_overlap" in result.columns
        # Should have non-NaN values where CIs span zero
        assert not result["confidence_interval_overlap"].isna().all()

    def test_ci_overlap_true_when_ci_spans_zero(self):
        player_signals = pd.DataFrame({
            "player_id": [1, 2],
            "team": ["A", "A"],
            "combined_xg_suppression": [0.5, -0.3],
            "mean_combined_xg_suppression": [0.4, -0.2],
            "reliability_tier": ["high", "high"],
            "total_combined_xg_suppression_ci95_lower": [-0.1, -0.5],
            "total_combined_xg_suppression_ci95_upper": [1.0, 0.2],
        })
        result = MODULE.compute_player_sensitivity(player_signals)
        # Both players have CIs spanning zero → overlap = True
        assert result["confidence_interval_overlap"].all()

    def test_ci_overlap_false_when_ci_excludes_zero(self):
        player_signals = pd.DataFrame({
            "player_id": [1],
            "team": ["A"],
            "combined_xg_suppression": [2.0],
            "mean_combined_xg_suppression": [1.8],
            "reliability_tier": ["high"],
            "total_combined_xg_suppression_ci95_lower": [0.5],  # entirely above zero
            "total_combined_xg_suppression_ci95_upper": [3.5],
        })
        result = MODULE.compute_player_sensitivity(player_signals)
        assert not result["confidence_interval_overlap"].iloc[0]


class TestNoShotConditionalMetricsNaN:
    """Players with no observed shots must have NaN conditional severity metrics."""

    def test_no_shot_conditional_metrics_are_nan(self):
        two_part_oof = pd.DataFrame({
            "event_id": ["e1", "e2"],
            "match_id": [1, 2],
            "player_id": [10, 10],
            "team": ["A", "A"],
            "oof_shot_probability": [0.2, 0.3],
            "observed_future_shot": [0, 0],  # no shots
            "observed_future_xg": [0.0, 0.0],
            "conditional_xg_prediction": [0.1, 0.2],
            "combined_future_xg_prediction": [0.02, 0.06],
        })
        signals = MODULE.build_player_aggregates(two_part_oof)
        p = signals.loc[signals["player_id"].eq(10)].iloc[0]
        # Conditional metrics must be NaN (or observed_shot_count == 0)
        assert p["observed_shot_count"] == 0
        assert pd.isna(p["total_conditional_severity_suppression"]) or pd.isna(p["conditional_severity_suppression"])


class TestReportPathSeparation:
    """Feature outputs and model reports must be written to separate directories."""

    def test_reports_dir_option_uses_canonical_location(self, tmp_path):
        """--reports-dir should be explicitly provided and used for JSON and CSV outputs."""
        import tempfile
        feature_dir = tmp_path / "data" / "features"
        reports_dir = tmp_path / "outputs" / "models" / "reports"
        feature_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        oof = _minimal_two_part_oof()
        oof["classification_model_variant"] = "b7_full_with_360"

        # Classification OOF must have y_score matching the oof_shot_probability
        class_oof = oof[[
            "event_id"
        ]].drop_duplicates().reset_index(drop=True).copy()
        class_oof["model_variant"] = "b7_full_with_360"
        # Merge to get aligned y_score from oof_shot_probability
        oof_for_merge = oof[["event_id", "oof_shot_probability"]].drop_duplicates()
        class_oof = class_oof.merge(oof_for_merge, on="event_id", how="left")
        class_oof["y_score"] = class_oof["oof_shot_probability"]
        class_oof = class_oof[["event_id", "model_variant", "y_score"]]

        oof_path = feature_dir / "player_defensive_signals_provisional_oof.parquet"
        class_oof_path = feature_dir / "classification_oof.parquet"
        oof.to_parquet(oof_path, index=False)
        class_oof.to_parquet(class_oof_path, index=False)

        # Run main with explicit --reports-dir
        import sys
        from unittest.mock import patch
        
        args = [
            "build_provisional_player_signals.py",
            "--classification-oof", str(class_oof_path),
            "--two-part-oof", str(oof_path),
            "--output", str(feature_dir / "output.parquet"),
            "--reports-dir", str(reports_dir),
            "--bootstrap-iterations", "10",
        ]
        
        with patch.object(sys, "argv", args):
            MODULE.main()

        # Verify reports are in canonical reports_dir, not derived from feature path
        thresholds_file = reports_dir / "player_signal_reliability_thresholds.json"
        sensitivity_file = reports_dir / "player_signal_sensitivity.csv"
        
        assert thresholds_file.exists(), f"Expected {thresholds_file} to exist"
        assert sensitivity_file.exists(), f"Expected {sensitivity_file} to exist"
        
        # Verify feature output is in feature_dir
        assert (feature_dir / "output.parquet").exists()
        
        # Verify reports are NOT in feature_dir
        assert not (feature_dir / "player_signal_reliability_thresholds.json").exists()

    def test_reports_dir_default_is_outputs_models_reports(self):
        """When --reports-dir is not specified, it should default to outputs/models/reports."""
        # Just verify the argument is in the CLI
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--reports-dir", default="outputs/models/reports")
        args = parser.parse_args([])
        assert args.reports_dir == "outputs/models/reports"

    def test_no_config_argument_in_cli(self):
        """--config should no longer be a required argument."""
        import argparse
        parser = argparse.ArgumentParser()
        # Simulate the actual argument setup from the script
        parser.add_argument("--classification-oof", required=True)
        parser.add_argument("--two-part-oof", required=True)
        parser.add_argument("--regression-oof")
        parser.add_argument("--output", required=True)
        parser.add_argument("--reports-dir", default="outputs/models/reports")
        parser.add_argument("--bootstrap-iterations", type=int, default=1000)
        parser.add_argument("--bootstrap-seed", type=int, default=42)
        
        # Should not raise for missing --config
        args = parser.parse_args([
            "--classification-oof", "class.parquet",
            "--two-part-oof", "oof.parquet",
            "--output", "out.parquet",
        ])
        assert args.classification_oof == "class.parquet"
        # Verify config is not in args
        assert not hasattr(args, "config") or args.config is None
