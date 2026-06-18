"""Tests for two-part hurdle future-xG modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dax.models.two_part_xg import (
    build_combined_oof,
    build_conditional_regressor,
    compute_hurdle_metrics,
    ConditionalModelSpec,
    filter_classification_oof,
    fit_conditional_fold,
    decile_ranking,
)


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






