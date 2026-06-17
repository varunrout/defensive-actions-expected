# Baseline Modeling — Progress Tracker

**Strategy:** [BASELINE_MODELING_STRATEGY.md](BASELINE_MODELING_STRATEGY.md)  
**Quick Ref:** [MODELING_QUICK_REFERENCE.md](MODELING_QUICK_REFERENCE.md)

---

## Phase 1: Data QA & Feature Profiling ✅ COMPLETE

- [x] Build `player_defensive_actions.parquet` with 360 support features
- [x] Profile feature distributions and missingness
- [x] Validate phase labels and target parity
- [x] Generate visual diagnostics (heatmaps, scatter plots, correlations)

**Outputs:**
- ✅ `data/features/player_defensive_actions.parquet`
- ✅ `outputs/validation/analysis/player_features/figures/*.png` (diagnostic plots)
- ✅ `notebooks/07_player_feature_audit.ipynb`
- ✅ `notebooks/08_player_feature_visualization.ipynb`

---

## Phase 2: Baseline Model Training ⏳ IN PROGRESS

### 2.1 Create training scripts
- [ ] `scripts/models/__init__.py` (package init)
- [x] `scripts/models/train_baseline_logistic.py`
  - [x] Load `player_defensive_actions.parquet`
  - [x] Feature preprocessing (categorical encoding, standardization)
  - [x] Train V0 (phase-only) with GroupKFold
  - [x] Train V1 (phase + spatial) with GroupKFold
  - [x] Train V2 (full features) with GroupKFold
  - [x] Save models to `outputs/models/baseline/`
  - [x] Save metrics to `outputs/validation/baseline/baseline_model_metrics.json`

### 2.2 Create evaluation script
- [x] `scripts/models/evaluate_baseline_model.py`
  - [x] Load trained models
  - [x] Generate ROC curves (one per model variant)
  - [x] Generate PR curves (one per model variant)
  - [ ] Generate calibration plots
  - [ ] Compute phase-stratified AUC
  - [x] Output feature importance rankings
  - [x] Save all plots to `outputs/validation/baseline/`

### 2.3 Expected outputs
- [x] `outputs/models/baseline/logistic_v0_phase_only.joblib`
- [x] `outputs/models/baseline/logistic_v1_spatial.joblib`
- [x] `outputs/models/baseline/logistic_v2_full_baseline.joblib`
- [x] `outputs/validation/baseline/baseline_model_metrics.json`
- [x] `outputs/validation/baseline/baseline_roc_curves.png`
- [x] `outputs/validation/baseline/baseline_pr_curves.png`
- [ ] `outputs/validation/baseline/baseline_calibration_plots.png`
- [x] `outputs/validation/baseline/baseline_feature_importance.png`
- [ ] `outputs/validation/baseline/baseline_phase_stratified_metrics.csv`

### 2.4 Football validation checks
- [ ] High press predicts higher threat than low block
- [ ] Actions closer to goal predict higher threat
- [ ] Negative support balance predicts higher threat
- [ ] Central lane actions predict higher threat
- [ ] Results documented in validation report

---

## Phase 3: Tree-Based Comparison 📋 PLANNED

### 3.1 Create training script
- [ ] `scripts/models/train_tree_baseline.py`
  - [ ] Train LightGBM model (V4)
  - [ ] Use same GroupKFold splits as logistic
  - [ ] Save model to `outputs/models/tree/`

### 3.2 Create comparison analysis
- [ ] Compare V4 AUC/AP vs V2 logistic
- [ ] Compute SHAP values for feature importance
- [ ] Identify key feature interactions
- [ ] Quantify non-linearity benefit

### 3.3 Expected outputs
- [ ] `outputs/models/tree/lightgbm_v4_baseline.joblib`
- [ ] `outputs/validation/tree/tree_vs_logistic_comparison.json`
- [ ] `outputs/validation/tree/shap_summary_plot.png`
- [ ] `outputs/validation/tree/shap_dependence_plots.png`

---

## Phase 4: Model Interpretation & Insights 📋 PLANNED

### 4.1 Create analysis scripts
- [ ] `scripts/analysis/compute_player_dax_scores.py`
  - [ ] Load best model
  - [ ] Compute predicted probabilities for all actions
  - [ ] Aggregate by player (mean, median, sum)
  - [ ] Rank players by DAx score
  - [ ] Save to `data/features/player_dax_scores.parquet`

- [ ] `scripts/analysis/marginal_effects_analysis.py`
  - [ ] Compute marginal effects for key features
  - [ ] Generate "what-if" scenarios
  - [ ] Create marginal effect plots

### 4.2 Expected outputs
- [ ] `data/features/player_dax_scores.parquet`
- [ ] `outputs/validation/analysis/player_dax_rankings_top50.csv`
- [ ] `outputs/validation/analysis/marginal_effects_plots.png`
- [ ] `docs/analysis/baseline_model_interpretation.md`

### 4.3 Validation checks
- [ ] Top 20 players by DAx make football sense
- [ ] Phase-specific patterns align with tactics
- [ ] Tournament winners rank highly
- [ ] Known defensive specialists ranked appropriately

---

## Phase 5: Iteration & Refinement 📋 BACKLOG

### 5.1 Feature engineering improvements
- [ ] Add interaction terms based on SHAP insights
- [ ] Test temporal features (rolling averages)
- [ ] Experiment with target windows (5s, 15s)

### 5.2 Phase-specific models
- [ ] Train separate models for high press vs low block
- [ ] Compare phase-specific vs unified model
- [ ] Document phase-specific insights

### 5.3 Advanced attributions
- [ ] Develop counterfactual simulation approach
- [ ] Model teammate interaction effects
- [ ] Attribution framework for team-level DAx

---

## Current Sprint Tasks (Next 48 hours)

1. **Create `scripts/models/` directory structure**
2. **Implement `train_baseline_logistic.py`**
   - Feature preprocessing pipeline
   - GroupKFold cross-validation
   - V0, V1, V2 variant training
3. **Run initial training**
4. **Quick validation of AUC scores**
5. **Implement `evaluate_baseline_model.py`**
6. **Generate diagnostic plots**
7. **Document initial results**

---

## Success Metrics Checklist

### Technical Success
- [ ] ROC-AUC ≥ 0.70 on held-out matches
- [ ] Average Precision ≥ 0.18 (3× base rate)
- [ ] Calibration error < 0.02
- [ ] Feature importance aligns with football logic

### Football Validity
- [ ] Phase ordering validated (press > counterpress > transition > block)
- [ ] Spatial logic validated (central + near goal = higher threat)
- [ ] Support logic validated (outnumbered = higher threat)
- [ ] Position logic validated (context-appropriate predictions)

### Interpretability
- [ ] Clear narrative for stakeholders
- [ ] Player insights make football sense
- [ ] Tactical patterns revealed
- [ ] Actionable recommendations possible

---

## Blockers & Dependencies

- None currently identified
- All data and features ready
- Environment configured with sklearn, pandas, lightgbm

---

## Notes & Decisions

- **2026-06-10:** Baseline modeling strategy finalized
- **Decision:** Start with logistic regression for interpretability, then tree-based for performance comparison
- **Decision:** Use GroupKFold by match to prevent leakage
- **Decision:** Primary metric is ROC-AUC, secondary is Average Precision
- **Decision:** Defer phase-specific models until baseline established

---

**Last Updated:** June 10, 2026  
**Status:** Phase 2 (Training) ready to begin
