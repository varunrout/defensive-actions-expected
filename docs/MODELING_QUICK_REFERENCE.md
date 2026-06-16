# Baseline Modeling Strategy — Quick Reference

**Full document:** [BASELINE_MODELING_STRATEGY.md](BASELINE_MODELING_STRATEGY.md)  
**Progress tracker:** [MODELING_PROGRESS.md](MODELING_PROGRESS.md)

---

## The Question
**How much attacking threat does a defensive action suppress?**

## The Model
**Predict:** `target_shot_in_10s` (binary, 5.7% positive rate)  
**Approach:** Logistic regression → tree-based comparison  
**Unit:** Player defensive action with full tactical context

---

## Feature Groups (40+ features)

### A. Action Context
- Action type, family, phase (9 phases)
- Player position group
- Play pattern, counterpress flag

### B. Spatial (Location)
- Action x/y coordinates
- Nearest goal distance
- Pitch zone (9 zones)
- Central/wide lane flags

### C. Temporal (Possession)
- Possession progress ratio
- Seconds since possession start
- Phase transition count

### D. Support (360 Freeze-Frame)
- Support balance (5m, 10m)
- Teammate/opponent nearest distance
- Team shape spread & centroids
- Player counts & ratios

---

## Model Variants

| Variant | Features | Purpose |
|---------|----------|---------|
| **V0** | Phase only | Baseline floor |
| **V1** | Phase + spatial | Spatial value test |
| **V2** | All features | Full baseline (recommended) |
| **V3** | V2 + interactions | Phase × location effects |
| **V4** | LightGBM | Non-linear benchmark |

---

## Evaluation

**Method:** GroupKFold by match (K=5)  
**Metrics:**
- **ROC-AUC:** Target >0.70 (discrimination)
- **Average Precision:** Target >0.18 (imbalanced class)
- **Calibration:** Brier score, calibration plots
- **Phase-stratified AUC:** Separate by defensive phase

**Football validation:**
- High press > counterpress > transition > mid block > low block
- Central actions near goal = higher threat
- Outnumbered defenders = higher threat

---

## Success Criteria

✅ **ROC-AUC ≥ 0.70** on held-out matches  
✅ **Average Precision ≥ 0.18** (3× base rate)  
✅ **Calibrated predictions** (<2% error)  
✅ **Football logic validated** (phase/spatial/support ordering)  
✅ **Interpretable insights** (clear player DAx scores)

---

## Implementation

### Data
- **Input:** `data/features/player_defensive_actions.parquet`
- **Rows:** ~60,000 defensive actions
- **Matches:** 115 with 360 data (WC 2022, Euro 2024)
- **Players:** ~1,000 unique defenders

### Scripts to Create
```powershell
# Training
scripts/models/train_baseline_logistic.py
scripts/models/train_tree_baseline.py

# Evaluation
scripts/models/evaluate_baseline_model.py

# Analysis
scripts/analysis/compute_player_dax_scores.py
scripts/analysis/marginal_effects_analysis.py
```

### Outputs
```
outputs/models/baseline/
  └── logistic_v2_full_baseline.joblib

outputs/models/tree/
  └── lightgbm_v4_baseline.joblib

outputs/validation/baseline/
  ├── baseline_model_metrics.json
  ├── baseline_roc_curves.png
  └── baseline_feature_importance.png

outputs/validation/tree/
  └── shap_summary_plot.png

outputs/oof/baseline/
  └── baseline_oof_predictions.parquet

data/features/
  └── player_dax_scores.parquet
```

---

## Interpretation Example

**Hypothetical logistic coefficients:**
- `phase=high_press`: +0.45 → Higher threat context
- `phase=settled_low_block`: -0.30 → Lower threat context
- `nearest_goal_distance`: -0.15 → Farther from goal = less threat
- `freeze_support_balance_5m`: -0.10 → More teammates = less threat
- `is_central_lane`: +0.25 → Central actions = higher threat

**Football story:**
- CB clearing in low block, 40m from goal, 3 teammates nearby: ~2% shot probability
- Forward pressing high, 20m from goal, outnumbered: ~15% shot probability

---

## Next Steps

1. ✅ Feature engineering complete (`player_defensive_actions.parquet`)
2. ⏳ Train V0-V2 logistic models with GroupKFold
3. ⏳ Evaluate ROC/PR curves, calibration, feature importance
4. ⏳ Train tree-based model (V4) for comparison
5. ⏳ Compute SHAP values and marginal effects
6. ⏳ Generate player DAx scores
7. ⏳ Validate football logic and document results

---

**Status:** Phase 8 (Suppression Model) — Baseline strategy defined, training in progress  
**Last Updated:** June 10, 2026
