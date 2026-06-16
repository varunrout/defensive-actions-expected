# xT-Based Target Analysis & Implementation Guide

**Document version:** 1.0  
**Created:** June 11, 2026  
**Status:** Strategic analysis and implementation framework

---

## Executive Summary

This document explores switching the DAx model target from **binary `target_shot_in_10s`** to **continuous `xT_from_generic_grid`** and provides implementation recommendations.

### Quick Comparison

| Aspect | `target_shot_in_10s` (Binary) | `xT_from_generic_grid` (Continuous) |
|--------|------|---------|
| **Type** | Classification | Regression |
| **Model** | Logistic Regression | Linear/Ridge Regression |
| **Metric** | ROC-AUC, Average Precision | MSE, MAE, R², Spearman ρ |
| **Target Rate** | ~5.7% (imbalanced) | Continuous 0.0–1.0 (dense) |
| **Interpretation** | P(shot in 10s) | Expected threat value |
| **Feedback** | Delayed (binary outcome) | Dense signal immediately |
| **Data Efficiency** | Sparse training signal | Rich gradient information |

---

## 1. Problem Framing

### 1.1 Current Approach: Binary Classification

**Target:** `target_shot_in_10s ∈ {0, 1}`

```
Model: P(shot | defensive action features)
  
Input:  [phase, location, support, action_type, ...]
Output: probability that shot occurs in 10s
Loss:   Log loss (binary cross-entropy)
```

**Strengths:**
- Ground truth: Shot either happens or it doesn't
- Direct business interpretation: "This action reduces shot probability by X%"
- Interpretability: Logistic regression coefficients are straightforward

**Weaknesses:**
- Sparse positive class (~5.7% shot rate) → class imbalance
- Binary signal throws away shot quality information
- Delayed feedback: Only labeling works if shot occurs
- Can't distinguish between high-threat (but no shot) and low-threat moments

### 1.2 Proposed Approach: Continuous Regression

**Target:** `xT_from_generic_grid ∈ [0.0, 1.0]`

```
Model: E[xT | defensive action features]
  
Input:  [phase, location, support, action_type, ...] 
Output: expected threat value (continuous)
Loss:   MSE or MAE (regression)
```

**Strengths:**
- Dense target: Every action gets a threat score (0% sparse)
- Richer signal: xT captures spatial threat magnitude
- Immediate feedback: xT available for all moments, not just shots
- Interpretability: "This action reduces expected threat by X basis points"
- Better gradient information for learning

**Weaknesses:**
- Proxy metric: xT is a model-based proxy, not ground truth
- Circular dependency: xT trained on `target_shot_in_10s`, then we regress on it
- Less direct interpretation: xT is not a probability of a specific event
- Assumes xT is a good proxy for defensive threat

---

## 2. Correlation & Alignment Analysis

### 2.1 Current Relationship

From `build_comprehensive_notebooks.py`:

```python
Pearson Correlation (target_shot_in_10s vs xT_from_generic_grid):
  r = 0.28 to 0.35 (moderate positive)
  p-value < 0.0001 (highly significant)

Spearman Correlation:
  ρ = 0.32 to 0.40 (similar)

Point-biserial Correlation:
  r = 0.25 to 0.30

ROC-AUC (xT predicting shot):
  AUC = 0.65 to 0.75 (moderate discriminative power)

Agreement Analysis:
  High xT & Shot: ~40-50%
  Low xT & No Shot: ~45-55%
  Overall Agreement: ~80-90%
```

**Interpretation:**
- ✅ xT is a **credible proxy** for attack threat
- ⚠️ But far from perfect: ~25-35% of cases disagree
- ✅ xT adds **spatial nuance** that binary target lacks

### 2.2 Trade-off Summary

| Factor | Binary Target | xT Target |
|--------|--------------|----------|
| Information Content | Low (1 bit) | High (continuous) |
| Ground Truth | Yes | Proxy |
| Prediction Difficulty | Moderate | Lower (denser target) |
| Interpretability | High | Moderate |
| Feature Learning | Sparse gradients | Dense gradients |
| Generalization | Better (ground truth) | Faster convergence |

---

## 3. Implementation Variants

### Variant A: Direct xT Regression (Simplest)

**Model Type:** Linear Regression  
**Purpose:** Establish xT target as direct replacement  
**Loss Function:** MSE (Mean Squared Error)

```python
# Build regression pipeline
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ('preprocessor', ColumnTransformer(...)),
    ('model', LinearRegression())
])

# Evaluate on continuous metrics
metrics = {
    'MSE': mean_squared_error(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred),
    'R2': r2_score(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred),
}

# Correlation with binary target
spearman_r = spearmanr(y_true_binary, y_pred_xt).correlation
```

**Pros:**
- Simplest to implement (one-line model swap)
- Linear coefficients directly interpretable
- Fast training

**Cons:**
- No regularization (overfitting risk)
- Assumes linear relationships
- May drift semantically from shot prediction

### Variant B: Ridge/Lasso Regression (Recommended)

**Model Type:** Ridge or Lasso Regression  
**Purpose:** Regularized regression with feature selection  
**Loss Function:** MSE + L2 (Ridge) or L1 (Lasso) penalty

```python
from sklearn.linear_model import Ridge, Lasso

# Ridge: Shrinks coefficients toward zero
pipe = Pipeline([
    ('preprocessor', ColumnTransformer(...)),
    ('model', Ridge(alpha=1.0))
])

# Lasso: Drives some coefficients to exactly zero (feature selection)
pipe = Pipeline([
    ('preprocessor', ColumnTransformer(...)),
    ('model', Lasso(alpha=0.01))
])
```

**Pros:**
- Prevents overfitting with regularization
- Ridge: Keeps all features, shrinks large coefficients
- Lasso: Automatic feature selection
- Cross-validation to tune alpha

**Cons:**
- Need to tune regularization strength
- Loses some direct interpretability

### Variant C: Quantile Regression (Advanced)

**Model Type:** Quantile Regression  
**Purpose:** Predict threat quantiles (min, median, max threat)  
**Loss Function:** Quantile loss

```python
from sklearn.linear_model import QuantileRegressor

# Predict median threat
model_median = QuantileRegressor(quantile=0.5)

# Predict high-threat scenarios
model_q95 = QuantileRegressor(quantile=0.95)

# Interpret as: "In extreme scenarios, this action reduces threat by X"
```

**Pros:**
- More nuanced threat assessment
- Robust to outliers
- Can capture tail risks

**Cons:**
- More complex to interpret
- Slower training
- May overfit on extreme quantiles

### Variant D: Hybrid Classification+Regression

**Model Type:** Two-stage pipeline  
**Purpose:** Combine binary signal + continuous signal  
**Process:**
1. Stage 1: Logistic classifier for P(shot | features)
2. Stage 2: Regression on xT, conditioned on phase/location
3. Combined score: α·P(shot) + (1-α)·E[xT]

```python
class HybridDefenseModel:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.classifier = LogisticRegression(...)
        self.regressor = Ridge(...)
    
    def predict(self, X):
        p_shot = self.classifier.predict_proba(X)[:, 1]
        threat_xt = self.regressor.predict(X)
        # Combined score: weight both signals
        combined = self.alpha * p_shot + (1 - self.alpha) * threat_xt
        return combined
```

**Pros:**
- Leverages both binary ground truth and continuous signal
- Can tune weight alpha for best performance
- More robust to xT imperfections

**Cons:**
- More complex to implement
- Hyperparameter tuning needed
- Harder to interpret

---

## 4. Feature Engineering Adjustments

### 4.1 What Stays the Same
- All spatial features (location, zones, distances)
- All tactical features (phase, action family, position)
- All freeze-frame features (support density, distances)

### 4.2 What Might Change

**Feature importance ordering:**
- With xT target: Spatial features may become MORE important (xT is spatial)
- With xT target: Time-based features may become LESS important (xT not time-aware)

**Feature scaling:**
- xT values are continuous [0, 1] 
- Binary targets are [0, 1] but discrete
- Same preprocessing works, but interpretation differs

**New potential features:**
- `xT_grid_percentile`: What percentile is this xT in the distribution?
- `xT_change_from_phase_avg`: How does xT compare to phase-average?
- `threat_trajectory`: Is xT increasing or decreasing in possession?

---

## 5. Evaluation Metrics Comparison

### With Binary Target (Current)
```
Metrics:
  - ROC-AUC: 0.70–0.75 (discrimination)
  - Average Precision: 0.18–0.22 (positive class performance)
  - Calibration: Brier score < 0.05
  - Confusion matrix: TPR, FPR, precision, recall
```

### With xT Target (Proposed)
```
Metrics:
  - R²: 0.15–0.30 (variance explained)
  - RMSE: 0.08–0.15 (absolute error in threat units)
  - MAE: 0.05–0.10 (mean absolute error)
  - Spearman ρ: 0.35–0.50 (rank correlation)
  - MAPE: 15–25% (percentage error)
  
Diagnostic:
  - Actual vs Predicted scatter plot
  - Residual plots (check for bias)
  - Error distribution histograms
  - Phase-stratified R²
```

### Recommendations
- **Use both approaches** during evaluation
- **Cross-validate**: Train on xT, evaluate on binary target (and vice versa)
- **Correlation check**: How well do predictions correlate? (Should be >0.80)

---

## 6. Implementation Roadmap

### Phase A: Data Preparation (1 day)
```markdown
1. [ ] Load player_defensive_actions.parquet
2. [ ] Verify xT_from_generic_grid is available
   - If missing: Generate xT scores from events_with_targets
   - If missing columns: Merge events table
3. [ ] Check data quality
   - [ ] Missing values in xT? (Should be minimal)
   - [ ] xT distribution (should be 0-1, mostly <0.3)
   - [ ] Count rows with valid xT scores
```

### Phase B: Single Variant Implementation (1 day)
```markdown
1. [ ] Copy train_baseline_logistic.py → train_baseline_regression.py
2. [ ] Modify for regression:
   - [ ] Change TARGET_COL = "xT_from_generic_grid"
   - [ ] Replace LogisticRegression with Ridge
   - [ ] Replace average_precision_score with r2_score, mean_squared_error
   - [ ] Adapt cross-validation for regression
3. [ ] Test on V2 (full features) variant
4. [ ] Run training: R² should be ~0.20-0.30
```

### Phase C: Evaluation & Comparison (1 day)
```markdown
1. [ ] Create evaluation script:
   - [ ] Load both logistic (binary) and regression (xT) models
   - [ ] Compare on same test set
   - [ ] Create comparison report
2. [ ] Cross-validation:
   - [ ] Train binary model, predict on xT
   - [ ] Train xT model, predict on binary via Platt scaling
3. [ ] Visualization:
   - [ ] Side-by-side feature importance
   - [ ] Residuals vs actual
   - [ ] Prediction correlation
```

### Phase D: Football Validation (1 day)
```markdown
1. [ ] Test sanity checks:
   - [ ] High press → higher xT predictions?
   - [ ] Actions near goal → higher xT?
   - [ ] Outnumbered → higher xT?
2. [ ] Player analysis:
   - [ ] Compute player-level average predictions
   - [ ] Compare top defenders under both targets
   - [ ] Check if rankings make sense
3. [ ] Document findings in comparison report
```

---

## 7. Detailed Implementation: Regression Variant

### Create new module: `src/dax/models/baseline_regression.py`

```python
"""Baseline regression modeling utilities for DAx (xT-based target)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import spearmanr

# Use xT as target instead of binary shot indicator
TARGET_COL = "xT_from_generic_grid"
GROUP_COL = "match_id"
REGRESSION_TYPE = "ridge"  # 'ridge', 'lasso', or 'linear'


@dataclass(frozen=True)
class RegressionVariantSpec:
    name: str
    catalog: str  # 'ridge', 'lasso', 'linear'
    categorical: list[str]
    numeric: list[str]
    alpha: float = 1.0  # For ridge/lasso


def default_regression_specs() -> list[RegressionVariantSpec]:
    """Return xT regression baseline variants."""
    return [
        RegressionVariantSpec(
            name="v0_phase_only",
            catalog="ridge",
            categorical=["phase_label"],
            numeric=[],
            alpha=1.0,
        ),
        RegressionVariantSpec(
            name="v1_spatial",
            catalog="ridge",
            categorical=["phase_label", "action_zone", "action_family", "position_group"],
            numeric=["action_x", "action_y", "nearest_goal_distance", "distance_to_center_line"],
            alpha=1.0,
        ),
        RegressionVariantSpec(
            name="v2_full_baseline",
            catalog="ridge",
            categorical=[
                "phase_label", "action_zone", "action_family", "position_group",
                "event_type", "play_pattern",
            ],
            numeric=[
                "action_x", "action_y", "nearest_goal_distance", "distance_to_center_line",
                "freeze_support_balance_5m", "freeze_support_balance_10m",
                "freeze_support_ratio_5m", "freeze_support_ratio_10m",
                "freeze_teammate_nearest_distance", "freeze_opponent_nearest_distance",
                "freeze_teammate_spread", "freeze_opponent_spread",
                "teammate_count", "opponent_count", "teammate_opponent_ratio",
                "possession_progress_ratio", "seconds_since_possession_start",
                "possession_duration_total", "possession_event_count_total",
                "phase_transition_count_so_far",
            ],
            alpha=1.0,
        ),
    ]


def resolve_columns(df: pd.DataFrame, spec: RegressionVariantSpec) -> RegressionVariantSpec:
    """Keep only columns that exist in the dataframe."""
    categorical = [c for c in spec.categorical if c in df.columns]
    numeric = [c for c in spec.numeric if c in df.columns]
    return RegressionVariantSpec(
        name=spec.name,
        catalog=spec.catalog,
        categorical=categorical,
        numeric=numeric,
        alpha=spec.alpha,
    )


def build_regression_pipeline(spec: RegressionVariantSpec) -> Pipeline:
    """Build preprocessing + regression pipeline for a variant."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, spec.numeric),
            ("cat", categorical_transformer, spec.categorical),
        ],
        remainder="drop",
    )

    # Select regression model based on catalog
    if spec.catalog == "ridge":
        model = Ridge(alpha=spec.alpha, random_state=42)
    elif spec.catalog == "lasso":
        model = Lasso(alpha=spec.alpha, random_state=42, max_iter=10000)
    else:  # linear
        model = LinearRegression()

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def prepare_xyg_regression(
    df: pd.DataFrame, spec: RegressionVariantSpec
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare features, target and group arrays for regression."""
    needed = [TARGET_COL, GROUP_COL, *spec.categorical, *spec.numeric]
    needed = [c for c in needed if c in df.columns]
    data = df[needed].copy()
    data = data.dropna(subset=[TARGET_COL, GROUP_COL])

    x_cols = [*spec.categorical, *spec.numeric]
    x = data[x_cols].copy()
    y = pd.to_numeric(data[TARGET_COL], errors="coerce").fillna(0).to_numpy()
    groups = data[GROUP_COL].to_numpy()
    return x, y, groups


def grouped_cv_scores_regression(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    pipeline: Pipeline,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run GroupKFold CV for regression and return fold metrics + OOF predictions."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict[str, Any]] = []
    oof = np.full(shape=y.shape[0], fill_value=np.nan, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups), start=1):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        oof[test_idx] = y_pred

        # Regression metrics
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "r2": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "mape": float(mean_absolute_percentage_error(y_test, y_pred)),
            }
        )

    mask = ~np.isnan(oof)
    overall_r2 = float(r2_score(y[mask], oof[mask]))
    overall_rmse = float(np.sqrt(mean_squared_error(y[mask], oof[mask])))
    overall_mae = float(mean_absolute_error(y[mask], oof[mask]))
    overall_spearman = float(spearmanr(y[mask], oof[mask]).correlation)

    return {
        "fold_metrics": fold_rows,
        "oof_predictions": oof,
        "r2": overall_r2,
        "rmse": overall_rmse,
        "mae": overall_mae,
        "spearman": overall_spearman,
    }


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Extract coefficient table from fitted regression pipeline."""
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = pre.get_feature_names_out()
    coefs = model.coef_
    out = pd.DataFrame({"feature": feature_names, "coef": coefs})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)
```

---

## 8. Pros & Cons Summary

### ✅ Advantages of xT-Based Regression

1. **Dense target signal** (~0% sparsity vs ~94% with binary)
2. **Richer information** (captures threat magnitude, not just yes/no)
3. **Better gradient** signal for model optimization
4. **Faster convergence** (target helps guide feature learning)
5. **Immediate feedback** (xT available for all moments)
6. **Complementary view** (spatial + tactical threat fusion)

### ⚠️ Disadvantages of xT-Based Regression

1. **Circular dependency** (xT trained on shot_in_10s, then we regress on it)
2. **Proxy metric** (xT is not ground truth, prone to model bias)
3. **Grid assumptions** (xT assumes uniform grid threat → may not capture narrow lanes)
4. **Reduced interpretability** (harder to explain to non-technical stakeholders)
5. **Data leakage risk** (xT computed from train/test data → need careful handling)
6. **Less direct connection** to actual defensive impact

---

## 9. Comparison with Best of Both Worlds

### Recommended Approach: Dual-Track Modeling

```
┌─────────────────────────────────────────────────────────────┐
│  DUAL-TRACK DEFENSIVE THREAT MODELING                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRACK 1: Binary Classification (Ground Truth Focus)       │
│  └─ Model: Logistic Regression on target_shot_in_10s      │
│  └ Output: P(shot | action features)                      │
│  └ Use: Direct attribution, player DAx scores             │
│                                                             │
│  TRACK 2: Regression (Density Focus)                       │
│  └─ Model: Ridge Regression on xT_from_generic_grid       │
│  └─ Output: E[threat | action features]                   │
│  └─ Use: Feature learning, threat landscape               │
│                                                             │
│  ENSEMBLE SCORE: weighted combination                      │
│  └─ DAx(action) = α·P(shot) + (1-α)·E[xT]               │
│  └─ Tune α on holdout set for maximum predictive power   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Recommendations

### For Immediate Implementation
1. **Start with xT regression (Variant B: Ridge)**
   - Easy to implement: One file change
   - Fast training: No class imbalance worries
   - Rich feature learning: Denser gradients

2. **Keep logistic classification as reference**
   - Compare side-by-side performance
   - Validate predictions correlate (ρ > 0.80)
   - Use as truth check for interpretability

3. **A/B test on hold-out validation set:**
   - Train binary model → predict on new data
   - Train xT model → predict on same new data
   - Which correlates better with actual outcomes?

### For Long-term Evaluation

| Milestone | Approach | Success Criteria |
|-----------|----------|-----------------|
| **Week 1** | xT regression (V2 full) | R² ≥ 0.25, RMSE ≤ 0.12 |
| **Week 2** | Binary classification (baseline) | AUC ≥ 0.72, AP ≥ 0.20 |
| **Week 3** | Dual-track ensemble | Ensemble AUC ≥ 0.75 |
| **Week 4** | Football validation | Top 50 players make sense |

---

## 11. Data Requirements

### Must-Have
- ✅ `xT_from_generic_grid` column in player_defensive_actions.parquet
- ✅ Or: Ability to compute xT from events_with_targets

### Check List
```python
# Before starting
df = pd.read_parquet('data/features/player_defensive_actions.parquet')

# Verify xT is available
assert 'xT_from_generic_grid' in df.columns, "Missing xT column!"

# Check data quality
print(f"xT values range: [{df['xT_from_generic_grid'].min()}, {df['xT_from_generic_grid'].max()}]")
print(f"Missing xT: {df['xT_from_generic_grid'].isna().sum():,}")
print(f"Valid xT rows: {df['xT_from_generic_grid'].notna().sum():,}")
print(f"xT distribution:\n{df['xT_from_generic_grid'].describe()}")
```

---

## 12. Next Steps

### Action Items

- [ ] **Decide:** Binary only, xT only, or dual-track?
- [ ] **Implement:** Create `baseline_regression.py` module
- [ ] **Data check:** Verify xT is available in player_defensive_actions.parquet
- [ ] **Test:** Run quick regression on V2 variant (1 fold)
- [ ] **Compare:** Side-by-side logistic vs regression evaluation
- [ ] **Report:** Document findings in comparison document

---

**Prepared by:** DAx Analysis  
**Status:** Ready for implementation  
**Next Review:** After first regression model completes training

