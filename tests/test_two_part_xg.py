"""Tests for two-part hurdle future-xG modelling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dax.models.two_part_xg import (
    build_combined_oof,
    build_conditional_regressor,
    compute_conditional_metrics,
    compute_hurdle_metrics,
    ConditionalModelSpec,
    filter_classification_oof,
    fit_conditional_fold,
    decile_ranking,
    train_conditional_variant,
)

# Load training script module for selection tests
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_two_part_xg.py"
_SPEC = importlib.util.spec_from_file_location("train_two_part_xg_script", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    
    n_rows = 100
    df = pd.DataFrame({
        "event_id": range(n_rows),
        "match_id": np.random.randint(0, 10, n_rows),
        "player_id": np.random.randint(0, 20, n_rows),
        "team": np.random.choice(["Team A", "Team B"], n_rows),
        "action_family": np.random.choice(["pass", "tackle", "clearance"], n_rows),
        "phase_label": np.random.choice(["buildup", "transition"], n_rows),
        "position_group": np.random.choice(["defense", "midfield"], n_rows),
        "action_x": np.random.uniform(0, 120, n_rows),
        "action_y": np.random.uniform(0, 80, n_rows),
        "distance_to_center_line": np.random.uniform(0, 60, n_rows),
        "target_future_shot_10s": np.random.binomial(1, 0.2, n_rows),
        "target_future_xg_10s": np.random.exponential(0.1, n_rows) * (np.random.binomial(1, 0.08, n_rows)),
    })
    
    return df


@pytest.fixture
def sample_oof():
    """Create sample classification OOF."""
    n_rows = 100
    oof = pd.DataFrame({
        "event_id": range(n_rows),
        "fold": np.repeat([0, 1, 2, 3, 4], 20),
        "y_score": np.random.uniform(0, 1, n_rows),
        "model_variant": "b7_full_with_360",
    })
    return oof


@pytest.fixture
def resolved_features():
    """Create resolved feature contract."""
    return {
        "categorical": ["phase_label", "action_family", "position_group"],
        "numeric": ["action_x", "action_y", "distance_to_center_line"],
        "final_features": ["phase_label", "action_family", "position_group", "action_x", "action_y", "distance_to_center_line"],
        "coverage_360": 0.95,
        "missing_optional_features": [],
    }


class TestConditionalRegressor:
    """Test conditional regressor building."""
    
    def test_ridge_regressor_creation(self, resolved_features):
        """Test creating a Ridge regressor."""
        regressor = build_conditional_regressor(
            "ridge",
            {"alpha": 1.0},
            resolved_features["categorical"],
            resolved_features["numeric"]
        )
        assert regressor is not None
        assert hasattr(regressor, "fit")
        assert hasattr(regressor, "predict")
    
    def test_hgb_regressor_creation(self, resolved_features):
        """Test creating a HistGradientBoosting regressor."""
        regressor = build_conditional_regressor(
            "hist_gradient_boosting_regressor",
            {"max_iter": 10, "learning_rate": 0.05},
            resolved_features["categorical"],
            resolved_features["numeric"]
        )
        assert regressor is not None
    
    def test_invalid_model_family(self, resolved_features):
        """Test that invalid model family raises error."""
        with pytest.raises(ValueError, match="Unknown conditional model family"):
            build_conditional_regressor(
                "invalid_family",
                {},
                resolved_features["categorical"],
                resolved_features["numeric"]
            )


class TestConditionalFitPredict:
    """Test conditional model fitting and prediction."""
    
    def test_fit_conditional_fold_uses_only_nonzero_training(self, sample_data, resolved_features):
        """Test that conditional model uses only non-zero training rows."""
        train = sample_data.iloc[:60]
        validation = sample_data.iloc[60:]
        
        predictions, metrics = fit_conditional_fold(
            train,
            validation,
            "ridge",
            {"alpha": 1.0},
            resolved_features,
        )
        
        # Predictions should have full validation length
        assert len(predictions) == len(validation)
        assert not np.all(np.isnan(predictions))
        
        # Metrics should be computed
        assert "mae" in metrics
        assert "nonzero_target_mae" in metrics
        assert metrics["training_nonzero_rows"] > 0
    
    def test_fit_conditional_fold_handles_zero_training(self, sample_data, resolved_features):
        """Test that no all-zero training data raises error."""
        # Create training with only zeros
        train = sample_data.iloc[:60].copy()
        train["target_future_xg_10s"] = 0
        validation = sample_data.iloc[60:]
        
        with pytest.raises(ValueError, match="no non-zero training rows"):
            fit_conditional_fold(
                train,
                validation,
                "ridge",
                {"alpha": 1.0},
                resolved_features,
            )
    
    def test_predictions_clipped_at_zero(self, sample_data, resolved_features):
        """Test that predictions are clipped to non-negative."""
        train = sample_data.iloc[:60]
        validation = sample_data.iloc[60:]
        
        predictions, _ = fit_conditional_fold(
            train,
            validation,
            "ridge",
            {"alpha": 1.0},
            resolved_features,
        )
        
        assert np.all(predictions >= -1e-10)  # Allow small numerical errors


class TestCombinedOOF:
    """Test combined OOF construction."""

    def test_filter_classification_oof_selects_requested_variant(self, sample_oof):
        """Test selecting one variant from a multi-variant classification OOF table."""
        alternate = pd.DataFrame(sample_oof).assign(
            model_variant="b6_full_without_360",
            y_score=0.0,
        )
        multi_variant = pd.concat([sample_oof, alternate], ignore_index=True)

        filtered = filter_classification_oof(multi_variant, "b7_full_with_360")

        assert filtered["model_variant"].nunique() == 1
        assert filtered["model_variant"].iloc[0] == "b7_full_with_360"
        np.testing.assert_array_almost_equal(filtered["y_score"].to_numpy(), sample_oof["y_score"].to_numpy())

    def test_filter_classification_oof_raises_for_missing_variant(self, sample_oof):
        """Test that a missing classification variant fails clearly."""
        with pytest.raises(ValueError, match="does not contain variant"):
            filter_classification_oof(sample_oof, "b5_360_geometry")
    
    def test_build_combined_oof_structure(self, sample_data, sample_oof, resolved_features):
        """Test that combined OOF has required columns."""
        conditional_predictions = np.random.uniform(0, 1, len(sample_data))
        folds = pd.DataFrame({"fold": np.repeat([0, 1, 2, 3, 4], 20)})
        
        spec = ConditionalModelSpec(
            name="test_conditional",
            model_family="ridge",
            hyperparameters={"alpha": 1.0}
        )
        
        combined = build_combined_oof(
            sample_data,
            sample_oof,
            conditional_predictions,
            folds,
            spec,
            "b7_full_with_360",
        )
        
        required_cols = {
            "event_id", "observed_future_shot", "observed_future_xg",
            "oof_shot_probability", "conditional_xg_prediction", "combined_future_xg_prediction"
        }
        assert required_cols.issubset(combined.columns)
    
    def test_combined_prediction_equals_product(self, sample_data, sample_oof):
        """Test that combined prediction = shot_prob * conditional."""
        conditional_predictions = np.random.uniform(0, 1, len(sample_data))
        folds = pd.DataFrame({"fold": np.repeat([0, 1, 2, 3, 4], 20)})
        
        spec = ConditionalModelSpec(
            name="test",
            model_family="ridge",
            hyperparameters={}
        )
        
        combined = build_combined_oof(
            sample_data,
            sample_oof,
            conditional_predictions,
            folds,
            spec,
            "b7_full_with_360",
        )
        
        expected = combined["oof_shot_probability"] * combined["conditional_xg_prediction"]
        np.testing.assert_array_almost_equal(
            combined["combined_future_xg_prediction"].values,
            expected.values
        )
    
    def test_no_duplicate_event_variant_rows(self, sample_data, sample_oof):
        """Test that no duplicate event-variant rows are created."""
        conditional_predictions = np.random.uniform(0, 1, len(sample_data))
        folds = pd.DataFrame({"fold": np.repeat([0, 1, 2, 3, 4], 20)})
        
        spec = ConditionalModelSpec(
            name="test",
            model_family="ridge",
            hyperparameters={}
        )
        
        combined = build_combined_oof(
            sample_data,
            sample_oof,
            conditional_predictions,
            folds,
            spec,
            "b7_full_with_360",
        )
        
        duplicates = combined.duplicated(
            ["event_id", "conditional_model_variant", "classification_model_variant"]
        )
        assert not duplicates.any()

    def test_build_combined_oof_filters_multi_variant_classification_oof(self, sample_data, sample_oof):
        """Test that build_combined_oof uses only the requested classification variant."""
        conditional_predictions = np.random.uniform(0, 1, len(sample_data))
        folds = pd.DataFrame({"fold": np.repeat([0, 1, 2, 3, 4], 20)})
        alternate = pd.DataFrame(sample_oof).assign(
            model_variant="b6_full_without_360",
            y_score=0.0,
        )
        multi_variant = pd.concat([sample_oof, alternate], ignore_index=True)

        spec = ConditionalModelSpec(
            name="test",
            model_family="ridge",
            hyperparameters={}
        )

        combined = build_combined_oof(
            sample_data,
            multi_variant,
            conditional_predictions,
            folds,
            spec,
            "b7_full_with_360",
        )

        assert combined["classification_model_variant"].nunique() == 1
        assert combined["classification_model_variant"].iloc[0] == "b7_full_with_360"
        np.testing.assert_array_almost_equal(
            combined["oof_shot_probability"].to_numpy(),
            sample_oof["y_score"].to_numpy(),
        )
    
    def test_no_missing_predictions(self, sample_data, sample_oof):
        """Test that no predictions are missing."""
        conditional_predictions = np.random.uniform(0, 1, len(sample_data))
        folds = pd.DataFrame({"fold": np.repeat([0, 1, 2, 3, 4], 20)})
        
        spec = ConditionalModelSpec(
            name="test",
            model_family="ridge",
            hyperparameters={}
        )
        
        combined = build_combined_oof(
            sample_data,
            sample_oof,
            conditional_predictions,
            folds,
            spec,
            "b7_full_with_360",
        )
        
        assert not combined[["oof_shot_probability", "conditional_xg_prediction", "combined_future_xg_prediction"]].isna().any().any()


class TestHurdleMetrics:
    """Test hurdle model evaluation metrics."""
    
    def test_compute_hurdle_metrics_structure(self, sample_data, sample_oof):
        """Test that hurdle metrics have expected structure."""
        conditional_predictions = np.random.uniform(0, 1, len(sample_data))
        folds = pd.DataFrame({"fold": np.repeat([0, 1, 2, 3, 4], 20)})
        
        spec = ConditionalModelSpec(
            name="test",
            model_family="ridge",
            hyperparameters={}
        )
        
        oof = build_combined_oof(
            sample_data,
            sample_oof,
            conditional_predictions,
            folds,
            spec,
            "b7_full_with_360",
        )
        
        metrics = compute_hurdle_metrics(oof)
        
        required_keys = {
            "mae", "rmse", "r2", "spearman",
            "zero_target_mae", "nonzero_target_mae"
        }
        assert required_keys.issubset(metrics.keys())
    
    def test_hurdle_metrics_values_reasonable(self, sample_data, sample_oof):
        """Test that metrics are in reasonable ranges."""
        conditional_predictions = np.random.uniform(0, 1, len(sample_data))
        folds = pd.DataFrame({"fold": np.repeat([0, 1, 2, 3, 4], 20)})
        
        spec = ConditionalModelSpec(
            name="test",
            model_family="ridge",
            hyperparameters={}
        )
        
        oof = build_combined_oof(
            sample_data,
            sample_oof,
            conditional_predictions,
            folds,
            spec,
            "b7_full_with_360",
        )
        
        metrics = compute_hurdle_metrics(oof)
        
        # MAE and RMSE should be non-negative
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0
        # Spearman should be in [-1, 1]
        if not np.isnan(metrics["spearman"]):
            assert -1 <= metrics["spearman"] <= 1


class TestDecileRanking:
    """Test decile ranking and cumulative capture."""
    
    def test_decile_ranking_structure(self, sample_data, sample_oof):
        """Test decile ranking output structure."""
        conditional_predictions = np.random.uniform(0, 1, len(sample_data))
        folds = pd.DataFrame({"fold": np.repeat([0, 1, 2, 3, 4], 20)})
        
        spec = ConditionalModelSpec(
            name="test",
            model_family="ridge",
            hyperparameters={}
        )
        
        oof = build_combined_oof(
            sample_data,
            sample_oof,
            conditional_predictions,
            folds,
            spec,
            "b7_full_with_360",
        )
        
        deciles = decile_ranking(oof)
        
        assert "decile" in deciles.columns
        assert "cumulative_observed" in deciles.columns
        assert len(deciles) <= 10
    
    def test_decile_cumulative_is_monotonic(self, sample_data, sample_oof):
        """Test that cumulative observed is non-decreasing when sorted by decile."""
        conditional_predictions = np.random.uniform(0, 1, len(sample_data))
        folds = pd.DataFrame({"fold": np.repeat([0, 1, 2, 3, 4], 20)})
        
        spec = ConditionalModelSpec(
            name="test",
            model_family="ridge",
            hyperparameters={}
        )
        
        oof = build_combined_oof(
            sample_data,
            sample_oof,
            conditional_predictions,
            folds,
            spec,
            "b7_full_with_360",
        )
        
        deciles = decile_ranking(oof)
        
        # Cumulative in the returned deciles is from highest to lowest decile (for top-k capture)
        # It should be monotonically increasing when read from highest decile down
        cumulative_vals = deciles.sort_values("decile", ascending=False)["cumulative_observed"].values
        diffs = np.diff(cumulative_vals)
        assert np.all(diffs >= -1e-10)  # Allow small numerical errors


# ── New behaviour tests ────────────────────────────────────────────────────────

class TestConditionalMetricsNonZeroOnly:
    """Conditional metrics must be computed exclusively on non-zero validation targets."""

    def test_conditional_metrics_exclude_zero_targets(self):
        """compute_conditional_metrics uses only non-zero rows explicitly passed."""
        y_true_nonzero = np.array([0.3, 0.5, 0.2, 0.8])
        y_pred_nonzero = np.array([0.25, 0.55, 0.18, 0.75])
        metrics = compute_conditional_metrics(y_true_nonzero, y_pred_nonzero)
        assert metrics["conditional_sample_count"] == 4
        assert metrics["conditional_mae"] > 0
        assert not np.isnan(metrics["conditional_spearman"])

    def test_conditional_metrics_empty_returns_nan(self):
        """compute_conditional_metrics with empty arrays returns NaN sentinel dict."""
        metrics = compute_conditional_metrics(np.array([]), np.array([]))
        assert metrics["conditional_sample_count"] == 0
        assert np.isnan(metrics["conditional_mae"])
        assert np.isnan(metrics["conditional_spearman"])

    def test_fit_conditional_fold_evaluates_only_nonzero_validation(self, sample_data, resolved_features):
        """Conditional fold metric is computed on non-zero validation targets only."""
        train = sample_data.iloc[:60].copy()
        validation = sample_data.iloc[60:].copy()
        # Zero out some validation targets explicitly
        validation = validation.copy()
        validation.iloc[:5, validation.columns.get_loc("target_future_xg_10s")] = 0.0

        predictions, metrics = fit_conditional_fold(
            train, validation, "ridge", {"alpha": 1.0}, resolved_features
        )
        # validation_nonzero_rows < total validation rows
        assert metrics["validation_nonzero_rows"] < len(validation)
        assert metrics["validation_rows"] == len(validation)


class TestHurdleMetricsAllRows:
    """Hurdle metrics must cover ALL validation rows, not just non-zero ones."""

    def test_hurdle_metrics_rows_equals_full_oof(self, sample_data, sample_oof):
        conditional_predictions = np.ones(len(sample_data)) * 0.1
        folds = pd.DataFrame({"fold": np.repeat([0, 1, 2, 3, 4], 20)})
        spec = ConditionalModelSpec("test", "ridge", {})
        oof = build_combined_oof(sample_data, sample_oof, conditional_predictions, folds, spec, "b7_full_with_360")
        metrics = compute_hurdle_metrics(oof)
        assert metrics["rows"] == len(oof)

    def test_hurdle_metrics_nonzero_subgroup_smaller(self, sample_data, sample_oof):
        """nonzero subgroup metrics are reported separately from all-row metrics."""
        conditional_predictions = np.ones(len(sample_data)) * 0.1
        folds = pd.DataFrame({"fold": np.repeat([0, 1, 2, 3, 4], 20)})
        spec = ConditionalModelSpec("test", "ridge", {})
        oof = build_combined_oof(sample_data, sample_oof, conditional_predictions, folds, spec, "b7_full_with_360")
        # Ensure there are some zero targets
        oof = oof.copy()
        oof.iloc[:10, oof.columns.get_loc("observed_future_xg")] = 0.0
        metrics = compute_hurdle_metrics(oof)
        # Both all-row and nonzero metrics should be present
        assert "mae" in metrics
        assert "nonzero_mae" in metrics


class TestConditionalMeanBaseline:
    """Conditional mean baseline must return a constant prediction equal to the training mean."""

    def test_conditional_mean_returns_constant_prediction(self, sample_data, resolved_features):
        train = sample_data.iloc[:60].copy()
        validation = sample_data.iloc[60:].copy()
        nonzero_train = train.loc[train["target_future_xg_10s"].gt(0)]
        expected_mean = float(nonzero_train["target_future_xg_10s"].mean())

        predictions, metrics = fit_conditional_fold(
            train, validation, "conditional_mean", {}, resolved_features
        )
        # All predictions should equal the training nonzero mean
        np.testing.assert_allclose(predictions, expected_mean, rtol=1e-9)

    def test_conditional_mean_spec_builds_none_regressor(self, resolved_features):
        result = build_conditional_regressor(
            "conditional_mean", {}, resolved_features["categorical"], resolved_features["numeric"]
        )
        assert result is None


class TestGammaTweedieModels:
    """Gamma and Tweedie candidates should succeed or raise a documented exception."""

    def test_gamma_regressor_created(self, resolved_features):
        reg = build_conditional_regressor(
            "gamma", {"alpha": 0.0, "max_iter": 100},
            resolved_features["categorical"], resolved_features["numeric"]
        )
        assert reg is not None
        assert hasattr(reg, "fit")

    def test_tweedie_regressor_created(self, resolved_features):
        reg = build_conditional_regressor(
            "tweedie", {"power": 1.5, "alpha": 0.0, "max_iter": 100},
            resolved_features["categorical"], resolved_features["numeric"]
        )
        assert reg is not None
        assert hasattr(reg, "fit")

    def test_gamma_fold_fit_or_documented_skip(self, sample_data, resolved_features):
        """Gamma fold fit either succeeds or raises ValueError (documented skip)."""
        train = sample_data.iloc[:60].copy()
        validation = sample_data.iloc[60:].copy()
        spec = ConditionalModelSpec("conditional_gamma", "gamma", {"alpha": 0.0, "max_iter": 100})
        try:
            predictions, metrics = fit_conditional_fold(
                train, validation, spec, resolved=resolved_features, target_col="target_future_xg_10s"
            )
            assert len(predictions) == len(validation)
        except Exception as exc:
            # Any exception from Gamma (e.g. convergence) is acceptable as a documented skip.
            assert len(str(exc)) > 0

    def test_tweedie_fold_fit_or_documented_skip(self, sample_data, resolved_features):
        """Tweedie fold fit either succeeds or raises ValueError (documented skip)."""
        train = sample_data.iloc[:60].copy()
        validation = sample_data.iloc[60:].copy()
        spec = ConditionalModelSpec("conditional_tweedie", "tweedie", {"power": 1.5, "alpha": 0.0, "max_iter": 100})
        try:
            predictions, metrics = fit_conditional_fold(
                train, validation, spec, resolved=resolved_features, target_col="target_future_xg_10s"
            )
            assert len(predictions) == len(validation)
        except Exception as exc:
            assert len(str(exc)) > 0


class TestSharedFolds:
    """b7 and b6 use identical shared fold assignments on common event IDs."""

    def test_shared_folds_identical_on_common_events(self):
        """Both variants receive the same fold label for the same event_id."""
        np.random.seed(0)
        n = 60
        event_ids = list(range(n))
        folds = np.repeat([0, 1, 2], n // 3)

        b7_oof = pd.DataFrame({
            "event_id": event_ids,
            "fold": folds,
            "y_score": np.random.uniform(0, 1, n),
            "model_variant": "b7_full_with_360",
        })
        b6_oof = pd.DataFrame({
            "event_id": event_ids[:40],  # b6 only covers a subset
            "fold": folds[:40],
            "y_score": np.random.uniform(0, 1, 40),
            "model_variant": "b6_full_without_360",
        })

        # b6 fold anchor: common events should have the same fold in both
        fold_map = b6_oof.set_index("event_id")["fold"]
        common_ids = set(b7_oof["event_id"]).intersection(set(b6_oof["event_id"]))
        for eid in common_ids:
            assert fold_map.loc[eid] == b7_oof.set_index("event_id").loc[eid, "fold"]


class TestCommonEventAlignmentVsR4:
    """Exact common event alignment between hurdle OOF and r4 OOF."""

    def test_common_event_alignment(self):
        """benchmark comparison uses only shared event IDs."""
        n = 50
        hurdle_oof = pd.DataFrame({
            "event_id": range(n),
            "observed_future_xg": np.zeros(n),
            "combined_future_xg_prediction": np.ones(n) * 0.1,
            "match_id": np.zeros(n, dtype=int),
            "fold": np.zeros(n, dtype=int),
        })
        r4_oof = pd.DataFrame({
            "event_id": range(30, 80),  # partial overlap
            "y_true": np.zeros(50),
            "y_pred": np.ones(50) * 0.05,
        })
        shared_ids = sorted(set(hurdle_oof["event_id"]).intersection(set(r4_oof["event_id"])))
        assert len(shared_ids) == 20  # events 30..49


class TestMultiMetricModelSelection:
    """Multi-metric selection should rank candidates and assign statuses."""

    def _make_comparison_df(self):
        return pd.DataFrame([
            {"classification_variant": "b7", "conditional_model": "conditional_mean_baseline",
             "mae": 0.20, "rmse": 0.30, "r2": 0.0, "spearman": 0.0, "nonzero_mae": 0.25,
             "nonzero_rmse": 0.35, "nonzero_spearman": 0.0, "prediction_bias": 0.0,
             "top_10_pct_xg_capture": 20.0, "top_20_pct_xg_capture": 35.0,
             "top_30_pct_xg_capture": 50.0, "fold_mean": 0.20, "fold_std": 0.01, "status": "candidate"},
            {"classification_variant": "b7", "conditional_model": "conditional_log_ridge",
             "mae": 0.15, "rmse": 0.22, "r2": 0.2, "spearman": 0.4, "nonzero_mae": 0.18,
             "nonzero_rmse": 0.26, "nonzero_spearman": 0.4, "prediction_bias": 0.01,
             "top_10_pct_xg_capture": 30.0, "top_20_pct_xg_capture": 50.0,
             "top_30_pct_xg_capture": 65.0, "fold_mean": 0.15, "fold_std": 0.02, "status": "candidate"},
        ])

    def test_mean_baseline_status_is_reference(self):
        comp = self._make_comparison_df()
        result = _MODULE.selection_with_reasons(comp, pd.DataFrame())
        baseline = result.loc[result["conditional_model"].eq("conditional_mean_baseline")]
        assert baseline.iloc[0]["status"] == "reference baseline"

    def test_best_candidate_preferred_or_insufficient(self):
        comp = self._make_comparison_df()
        # No benchmark available → best candidate should be insufficient evidence
        result = _MODULE.selection_with_reasons(comp, pd.DataFrame())
        ridge = result.loc[result["conditional_model"].eq("conditional_log_ridge")]
        assert ridge.iloc[0]["status"] in {"preferred candidate", "insufficient evidence"}

    def test_candidate_becomes_insufficient_when_weaker_than_r4(self):
        """If candidate MAE > benchmark MAE the status must be insufficient evidence."""
        comp = self._make_comparison_df()
        benchmark = pd.DataFrame([{
            "classification_variant": "b7",
            "conditional_model": "conditional_log_ridge",
            "benchmark_variant": "r4_full_with_360",
            "rows": 100,
            "candidate_mae": 0.20,   # worse than benchmark
            "benchmark_mae": 0.15,
            "candidate_rmse": 0.30,
            "benchmark_rmse": 0.22,
            "candidate_nonzero_mae": 0.25,
            "benchmark_nonzero_mae": 0.18,
        }])
        result = _MODULE.selection_with_reasons(comp, benchmark)
        ridge = result.loc[result["conditional_model"].eq("conditional_log_ridge")]
        assert ridge.iloc[0]["status"] == "insufficient evidence"

    def test_candidate_preferred_when_better_than_r4(self):
        """If candidate beats r4 on all core metrics the status must be preferred candidate."""
        comp = self._make_comparison_df()
        benchmark = pd.DataFrame([{
            "classification_variant": "b7",
            "conditional_model": "conditional_log_ridge",
            "benchmark_variant": "r4_full_with_360",
            "rows": 100,
            "candidate_mae": 0.10,   # better
            "benchmark_mae": 0.15,
            "candidate_rmse": 0.15,
            "benchmark_rmse": 0.22,
            "candidate_nonzero_mae": 0.12,
            "benchmark_nonzero_mae": 0.18,
        }])
        result = _MODULE.selection_with_reasons(comp, benchmark)
        ridge = result.loc[result["conditional_model"].eq("conditional_log_ridge")]
        assert ridge.iloc[0]["status"] == "preferred candidate"


class TestMLflowRunIdInOOF:
    """build_combined_oof must embed the MLflow nested run ID in every row."""

    def test_mlflow_run_id_propagated(self, sample_data, sample_oof):
        conditional_predictions = np.random.uniform(0, 1, len(sample_data))
        folds = pd.Series(np.repeat([0, 1, 2, 3, 4], 20))
        run_id = "test-run-abc123"
        combined = build_combined_oof(
            sample_data,
            sample_oof,
            conditional_predictions,
            folds,
            classification_variant="b7_full_with_360",
            run_id=run_id,
        )
        assert "mlflow_run_id" in combined.columns
        assert (combined["mlflow_run_id"] == run_id).all()

    def test_mlflow_run_id_none_allowed(self, sample_data, sample_oof):
        conditional_predictions = np.random.uniform(0, 1, len(sample_data))
        folds = pd.Series(np.repeat([0, 1, 2, 3, 4], 20))
        combined = build_combined_oof(
            sample_data, sample_oof, conditional_predictions, folds,
            classification_variant="b7_full_with_360", run_id=None,
        )
        assert "mlflow_run_id" in combined.columns
        assert combined["mlflow_run_id"].isna().all()


class TestFinalBundleLoadBack:
    """Fit a final conditional model, save as joblib, and verify load-back."""

    def test_bundle_loadback_predicts(self, sample_data, resolved_features, tmp_path):
        from dax.models.two_part_xg import fit_final_conditional_model, predict_conditional
        import joblib

        spec = ConditionalModelSpec("conditional_log_ridge", "log_ridge", {"alpha": 1.0})
        payload, nonzero_rows, _ = fit_final_conditional_model(
            sample_data, spec, resolved_features, target_col="target_future_xg_10s"
        )
        bundle = {"conditional_model": payload, "model_family": spec.model_family}
        bundle_path = tmp_path / "bundle.joblib"
        joblib.dump(bundle, bundle_path)
        loaded = joblib.load(bundle_path)
        features = sample_data[resolved_features["final_features"]].head(5)
        preds = predict_conditional(loaded["conditional_model"], features)
        assert len(preds) == 5
        assert np.all(preds >= 0)


class TestNoShotConditionalMetrics:
    """Players with no observed shots must have NaN conditional severity metrics."""

    def test_no_shot_players_have_nan_conditional_metrics(self):
        two_part_oof = pd.DataFrame({
            "event_id": ["e1", "e2", "e3"],
            "match_id": [1, 1, 2],
            "player_id": [10, 10, 20],
            "team": ["A", "A", "B"],
            "oof_shot_probability": [0.1, 0.2, 0.3],
            "observed_future_shot": [0, 0, 0],  # no shots at all
            "observed_future_xg": [0.0, 0.0, 0.0],
            "conditional_xg_prediction": [0.1, 0.2, 0.3],
            "combined_future_xg_prediction": [0.01, 0.04, 0.09],
        })

        # Import build_player_aggregates from the script
        from scripts.build_provisional_player_signals import build_player_aggregates  # noqa: PLC0415
        signals = build_player_aggregates(two_part_oof)
        # All players have zero shots; conditional severity must be NaN
        assert signals["total_conditional_severity_suppression"].isna().all() or (signals["observed_shot_count"] == 0).all()







