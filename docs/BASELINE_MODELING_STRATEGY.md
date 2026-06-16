# Baseline Modeling Strategy — Defensive Actions Expected (DAx)

**Document version:** 1.0  
**Last updated:** June 10, 2026  
**Status:** Strategic foundation for player defensive model development

**Quick reference:** [MODELING_QUICK_REFERENCE.md](MODELING_QUICK_REFERENCE.md) (1-page summary)

---

## Executive Summary

This document defines the baseline modeling strategy for the Defensive Actions Expected (DAx) project. The model measures defending as **attacking option suppression** by quantifying how defensive actions reduce the probability of dangerous attacking outcomes.

**Core modeling objective:**  
Predict whether a possession will lead to a shot within 10 seconds (`target_shot_in_10s`), conditional on:
- Player defensive actions
- Spatial context (location, goal distance, pitch zones)
- Temporal context (phase, possession progress)
- Tactical context (freeze-frame player support and pressure)

**Baseline approach:**  
Start with **logistic regression** on engineered features for interpretability, then compare against tree-based models (gradient boosting) if non-linear interactions prove critical.

---

## 1. Problem Framing

### 1.1 Football question
**How much attacking threat does a defensive action suppress?**

Traditional defensive metrics (tackles, interceptions, clearances) count events but miss:
- Lane-blocking that prevents passes before they're attempted
- Positional pressure that forces backwards play
- Support positioning that enables teammates to press safely
- Compact shape that removes central access

### 1.2 Statistical question
Given a defensive action in context, what is the **counterfactual change in shot probability**?

Formally:
```
DAx = P(shot | no defensive action) - P(shot | defensive action observed)
```

For the baseline model, we approximate this by:
1. Modeling `P(shot in 10s | defensive action features)`
2. Comparing predictions across action types, phases, and spatial contexts
3. Attributing threat reduction to individual players based on their action contexts

### 1.3 Modeling paradigm
- **Supervised classification:** Binary target (`target_shot_in_10s`)
- **Unit of analysis:** Player defensive action (one row per defensive event)
- **Evaluation:** Match-grouped cross-validation to avoid data leakage
- **Interpretation:** Feature importance and marginal effect analysis

---

## 2. Dataset & Features

### 2.1 Data sources
**Primary input:** `data/features/player_defensive_actions.parquet`

**Coverage:**
- 166 matches across 3 tournaments (Euro 2020, World Cup 2022, Euro 2024)
- ~368,000 events with 360 freeze-frame data (~60% of total events)
- Defensive actions only: Pressure, Ball Recovery, Duels, Clearances, Blocks, Interceptions, Fouls, GK actions, Shields
- Player-identifiable actions with full tactical context

**Current dataset:**
- Rows: ~40,000–80,000 defensive actions (estimated)
- Matches: 115 with 360 data (WC 2022: 64, Euro 2024: 51)
- Players: ~1,000+ unique defenders
- Positive rate: ~5.7% (shot in 10s)

### 2.2 Feature architecture

#### Group A: Action identifiers & context
- `event_type`: Pressure, Ball Recovery, Duel, etc.
- `action_family`: `pressure`, `recovery`, `contest`, `intervention`, `discipline`, `goalkeeper`, `possession_protection`
- `position_group`: Goalkeeper, Centre Back, Fullback/Wingback, Defensive Midfielder, Midfielder, Winger, Forward
- `phase_label`: 9 defensive phases (high press, counterpress, transition, settled mid/low block, box defense, wide defense, central progression, second ball)
- `play_pattern`: Regular play, counter-attack, etc.
- `counterpress`: Boolean flag

#### Group B: Spatial features (action location)
- `action_x`, `action_y`: Pitch coordinates (0-120m x 0-80m)
- `action_zone`: 9 pitch zones (defensive/middle/attacking third × left/center/right)
- `nearest_goal_distance`: Distance to nearest goal
- `distance_to_center_line`: Vertical distance from pitch center
- `is_central_lane`, `is_wide_lane`, `is_deep_zone`, `is_high_zone`: Boolean spatial flags

#### Group C: Temporal features (possession dynamics)
- `possession_progress_ratio`: How far through the possession (0–1)
- `seconds_since_possession_start`: Time elapsed in possession
- `possession_duration_total`: Total possession length
- `phase_transition_count_so_far`: Number of defensive phase changes
- `phase_changed_since_prev_event`: Boolean phase transition flag

#### Group D: Freeze-frame support features (360 data)
**Density at action location:**
- `freeze_support_balance_5m`: Teammates - opponents within 5m
- `freeze_support_balance_10m`: Teammates - opponents within 10m
- `freeze_support_ratio_5m`: Teammates / opponents within 5m
- `freeze_support_ratio_10m`: Teammates / opponents within 10m

**Nearest player distances:**
- `freeze_teammate_nearest_distance`: Distance to closest teammate
- `freeze_opponent_nearest_distance`: Distance to closest opponent

**Team shape metrics:**
- `freeze_teammate_centroid_x/y`: Average teammate position
- `freeze_opponent_centroid_x/y`: Average opponent position
- `freeze_teammate_spread`: Average distance of teammates from their centroid
- `freeze_opponent_spread`: Average distance of opponents from their centroid

**Counts:**
- `freeze_teammate_count`: Total teammates visible in 360 frame
- `freeze_opponent_count`: Total opponents visible in 360 frame
- `teammate_opponent_ratio`: Ratio of visible players

### 2.3 Target variable
**`target_shot_in_10s`:** Binary (0/1)
- 1 if the possession leads to a shot attempt within 10 seconds of the current event
- 0 otherwise
- Positive rate: ~5.7% (appropriate for imbalanced classification)
- Computed using forward-looking time window across possession boundaries

---

## 3. Baseline Model Architecture

### 3.1 Model selection rationale

**Primary baseline: Logistic Regression**
- **Why:** Maximum interpretability for football stakeholders
- **Coefficients** directly indicate feature importance and direction
- **Marginal effects** easily calculated for "what-if" scenarios
- **Regularization** (L2) prevents overfitting with correlated spatial features
- **Baseline performance** establishes minimum viable accuracy

**Secondary baseline: Gradient Boosting (LightGBM or XGBoost)**
- **Why:** Capture non-linear interactions between phase, location, and support
- **Tree-based** models handle feature interactions naturally
- **Comparison** against logistic regression quantifies non-linearity benefit
- **Feature importance** via SHAP values for validation

### 3.2 Feature engineering pipeline

**Preprocessing steps:**
1. **Handle missing values:**
   - Freeze-frame features: Missing = no 360 data → exclude row OR impute with phase-specific medians
   - Spatial features: Should never be missing for defensive actions (validation check)

2. **Categorical encoding:**
   - `action_family`: One-hot encoding (6–7 categories)
   - `position_group`: One-hot encoding (7–8 categories)
   - `phase_label`: One-hot encoding (9 categories)
   - `action_zone`: One-hot encoding (9 zones)

3. **Numeric standardization:**
   - Z-score normalization for continuous features (distances, counts, ratios)
   - Preserve interpretability by tracking means/stds for back-transformation

4. **Feature selection:**
   - Start with all features; use L1 regularization (Lasso) for automated selection
   - Validate importance via permutation importance
   - Remove features with near-zero variance or >0.95 correlation

### 3.3 Model variants

**V0: Phase-only baseline**
- Features: `phase_label` only
- Purpose: Establish phase-based threat rates as floor

**V1: Spatial baseline**
- Features: Phase + action location + goal distance + zones
- Purpose: Test spatial information value

**V2: Full baseline (recommended starting point)**
- Features: All groups (action, spatial, temporal, freeze-frame)
- Purpose: Full feature set with L2 regularization

**V3: Interaction-aware**
- Add engineered interactions:
  - `phase × action_family`
  - `phase × nearest_goal_distance`
  - `is_central_lane × freeze_support_balance_5m`
- Purpose: Capture tactical dependencies

**V4: Tree-based ensemble**
- Model: LightGBM with default hyperparameters
- Purpose: Non-linear benchmark

---

## 4. Evaluation Strategy

### 4.1 Cross-validation design

**GroupKFold by match (K=5)**
- **Why:** Prevent data leakage from same-match events appearing in train and test
- **Groups:** `match_id`
- **Folds:** 5-fold split (~33 matches per fold in test set)
- **Stratification:** Ensure balanced phase and shot-rate distribution across folds

### 4.2 Metrics

**Primary metrics:**
1. **ROC-AUC:** Discrimination ability across probability thresholds
   - Target: >0.65 (baseline), >0.70 (good), >0.75 (excellent)
2. **Average Precision (AP):** Performance on imbalanced positive class
   - Target: >0.15 (baseline at 5.7% base rate), >0.20 (good)

**Secondary metrics:**
3. **Calibration:** Brier score and calibration plots
   - Ensure predicted probabilities match observed frequencies
4. **Feature importance:** Permutation importance and SHAP values
5. **Phase-stratified AUC:** Separate performance by defensive phase

**Diagnostic metrics:**
6. **Confusion matrix at decision threshold** (e.g., 0.1 for high recall)
7. **Precision-recall curves**
8. **Lift curves** (compare to base rate)

### 4.3 Football validation

**Sanity checks:**
1. **Phase ordering:** High-press and counterpress should predict higher threat than settled low block
2. **Spatial logic:** Actions closer to goal should predict higher threat
3. **Support logic:** Negative support balance (outnumbered) should predict higher threat
4. **Position logic:** Forwards/attackers defending should predict higher threat than centre-backs in settled defense

**Case studies:**
- Extract top 10 highest-threat predictions → verify with video or event sequence
- Extract top 10 lowest-threat predictions → verify defensive solidity
- Compare model outputs to known tactical patterns (e.g., Guardiola high press vs. Mourinho low block)

---

## 5. Implementation Roadmap

### Phase 1: Data QA & Feature Profiling ✅ (COMPLETE)
- [x] Build `player_defensive_actions.parquet` with 360 support features
- [x] Profile feature distributions and missingness
- [x] Validate phase labels and target parity
- [x] Generate visual diagnostics (heatmaps, scatter plots, correlations)

### Phase 2: Baseline Model Training (CURRENT)
**Tasks:**
- [ ] Train V0-V2 logistic regression variants
- [ ] Evaluate with GroupKFold cross-validation
- [ ] Generate ROC curves, PR curves, calibration plots
- [ ] Compute feature importance (coefficients, permutation)
- [ ] Save trained models to `outputs/models/baseline/`

**Scripts to create:**
- `scripts/models/train_baseline_logistic.py`
- `scripts/models/evaluate_baseline_model.py`

**Outputs:**
- `outputs/models/baseline/logistic_v0_phase_only.joblib`
- `outputs/models/baseline/logistic_v2_full_baseline.joblib`
- `outputs/validation/baseline/baseline_model_metrics.json`
- `outputs/validation/baseline/baseline_roc_curves.png`
- `outputs/validation/baseline/baseline_feature_importance.png`

### Phase 3: Tree-Based Comparison
**Tasks:**
- [ ] Train LightGBM model (V4)
- [ ] Compare AUC/AP vs. logistic baseline
- [ ] Compute SHAP values for top features
- [ ] Identify key interactions (phase × location, support × pressure)

**Scripts:**
- `scripts/models/train_tree_baseline.py`

**Outputs:**
- `outputs/models/tree/lightgbm_v4_baseline.joblib`
- `outputs/validation/tree/tree_vs_logistic_comparison.json`
- `outputs/validation/tree/shap_summary_plot.png`

### Phase 4: Model Interpretation & Insights
**Tasks:**
- [ ] Generate phase-stratified model performance reports
- [ ] Compute marginal effects for key features (e.g., "What if support balance increases by 1 player?")
- [ ] Create player-level DAx scores by aggregating residuals or predicted probabilities
- [ ] Validate against football intuition and known player/team reputations

**Scripts:**
- `scripts/analysis/compute_player_dax_scores.py`
- `scripts/analysis/marginal_effects_analysis.py`

**Outputs:**
- `data/features/player_dax_scores.parquet`
- `outputs/validation/analysis/marginal_effects_plots.png`
- `docs/analysis/baseline_model_interpretation.md`

### Phase 5: Iteration & Refinement
**Tasks:**
- [ ] Add engineered interaction features based on SHAP insights
- [ ] Test phase-specific models (separate models for high press vs. low block)
- [ ] Experiment with temporal features (rolling averages, possession momentum)
- [ ] A/B test different target windows (5s, 10s, 15s)

---

## 6. Success Criteria

### 6.1 Technical success
- **ROC-AUC ≥ 0.70** on held-out matches
- **Average Precision ≥ 0.18** (3× improvement over base rate)
- **Calibration error < 0.02** (predictions match observed frequencies)
- **Feature importance aligns with football logic** (phase, location, support all significant)

### 6.2 Football validity
- **Phase ordering validated:** High press > counterpress > transition > mid block > low block (threat prediction order)
- **Spatial logic validated:** Central actions near goal predict higher threat
- **Support logic validated:** Outnumbered defenders face higher threat
- **Position logic validated:** Forwards defending ≠ centre-backs defending

### 6.3 Interpretability
- **Clear narrative:** "A defensive action in [phase] at [location] with [support context] reduces shot probability by [X]%"
- **Player insights:** Top 20 defenders by DAx score make football sense
- **Tactical insights:** Model reveals phase-specific defensive effectiveness patterns

---

## 7. Known Limitations & Mitigation

### 7.1 Data limitations
**Limitation:** 360 data only available for ~60% of events
- **Mitigation:** Restrict baseline model to 360-only actions; expand later with imputation or separate non-360 model

**Limitation:** 360 freeze-frames don't identify all visible players
- **Mitigation:** Use aggregate density and distance features, not individual player tracking

**Limitation:** Target is possession-level, not action-level
- **Mitigation:** Model predicts possession outcome, not direct causal effect of single action; interpret as "context threat level"

### 7.2 Modeling limitations
**Limitation:** Logistic regression assumes linear feature effects
- **Mitigation:** Compare against tree-based models; add polynomial/interaction terms if needed

**Limitation:** Action attribution is uncertain (team defense is collective)
- **Mitigation:** Model as "action context threat," not "action causal effect"; use SHAP for partial attribution

**Limitation:** Model cannot see defensive actions that prevented events from happening
- **Mitigation:** Accept as inherent data limitation; future work could explore counterfactual simulation

---

## 8. Next Steps (Immediate Actions)

1. **Create training script:** `scripts/models/train_baseline_logistic.py`
   - Load `player_defensive_actions.parquet`
   - Split features/target, encode categoricals, standardize numerics
   - Train V0, V1, V2 variants with GroupKFold by match
   - Save models and metrics

2. **Create evaluation script:** `scripts/models/evaluate_baseline_model.py`
   - Load trained models
   - Generate ROC/PR curves, calibration plots
   - Compute phase-stratified metrics
   - Output reports to `outputs/validation/baseline/`

3. **Run baseline training:**
   ```powershell
   .\.venv\Scripts\python.exe scripts\models\train_baseline_logistic.py
   .\.venv\Scripts\python.exe scripts\models\evaluate_baseline_model.py
   ```

4. **Document results:** Create `docs/analysis/baseline_model_results.md` with:
   - Performance summary table
   - Feature importance rankings
   - Football validation checks
   - Recommendations for next iteration

---

## 9. References

**Project documentation:**
- `docs/project_scope.md` — Overall DAx vision and phases
- `docs/player_defense_model.md` — Player-level feature design
- `docs/notebook_findings_summary.md` — Data quality validation

**Key notebooks:**
- `notebooks/05_threat_model_validation.ipynb` — Phase-level modeling benchmark
- `notebooks/07_player_feature_audit.ipynb` — Feature quality checks
- `notebooks/08_player_feature_visualization.ipynb` — Feature exploration

**Implementation resources:**
- `src/dax/features/player_defense.py` — Feature engineering module
- `scripts/features/build_player_defense_dataset.py` — Dataset builder
- `scripts/features/profile_player_defense_features.py` — Feature profiler

---

## 10. Appendix: Example Model Coefficients Interpretation

**Hypothetical logistic regression output (V2 model):**

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `phase_label=high_press` | +0.45 | High press actions are in higher-threat contexts |
| `phase_label=settled_low_block` | -0.30 | Low block actions are in lower-threat contexts |
| `nearest_goal_distance` | -0.15 | Each 10m farther from goal reduces log-odds by 1.5 |
| `freeze_support_balance_5m` | -0.10 | Each extra teammate within 5m reduces threat |
| `freeze_opponent_nearest_distance` | -0.20 | Closer opponent pressure increases threat |
| `is_central_lane` | +0.25 | Central defensive actions face higher threat |
| `action_family=pressure` | +0.10 | Pressure actions occur in slightly higher-threat moments |
| `action_family=clearance` | -0.05 | Clearances occur in slightly lower-threat moments (after danger passed) |

**Football interpretation:**
A centre-back making a clearance in a settled low block, 40m from goal, with 3 teammates within 5m, faces a predicted shot probability of ~2%. A forward pressing in the high press, 20m from goal, outnumbered by opponents, faces a predicted shot probability of ~15%.

---

**End of document**
