# Defensive Actions Expected - Implementation Plan

This file tracks practical delivery from MVP to advanced DAx.

## Key Documentation

- **Baseline modeling strategy:** `docs/BASELINE_MODELING_STRATEGY.md`
- Findings summary: `docs/notebook_findings_summary.md`
- Remediation plan: `docs/notebook_remediation_plan.md`

## Current status

- [x] Phase 1 framing translated into DAx formula and scope
- [x] Phase 2 baseline data ingestion for StatsBomb Open Data + 360 joins
- [x] Phase 3 rule-based phase segmentation
- [x] Phase 4 xT-style attacking threat baseline (`shot_in_10s` target)
- [x] Phase 6 player defensive features (feature engineering complete)
- [~] Phase 8 suppression model (baseline modeling strategy defined, training in progress)
- [ ] Phase 5 attacking option tree (deferred)
- [ ] Phase 7 team defensive features
- [ ] Phase 9 attribution framework
- [ ] Phase 10 phase-specific models
- [ ] Phase 11 validation suite and case studies
- [ ] Phase 12 dashboard and storytelling outputs
- [ ] Phase 13 portfolio packaging

**Current focus:** Baseline player defensive action model (logistic regression → tree-based comparison)

## MVP objective (now)

Build reproducible tables with these columns:

- `match_id`, `possession_id`, `event_id`, `timestamp`
- `team_in_possession`, `defending_team`
- ball location (`ball_x`, `ball_y`)
- `event_type`, `event_outcome`
- 360 freeze-frame counts and visibility flags
- defensive `phase_label`
- attacking baseline `threat_base_score`

## Next implementation step

**Baseline player defensive model (Phase 8 — in progress):**

1. Train logistic regression variants (V0: phase-only, V1: spatial, V2: full features)
2. Implement GroupKFold cross-validation by match to prevent leakage
3. Generate evaluation metrics: ROC-AUC, Average Precision, calibration plots
4. Compute feature importance (coefficients, permutation importance)
5. Compare against tree-based model (LightGBM) with SHAP analysis
6. Validate football logic: phase ordering, spatial logic, support dynamics
7. Output player-level DAx scores

**Scripts to create:**
- `scripts/models/train_baseline_logistic.py`
- `scripts/models/evaluate_baseline_model.py`
- `scripts/models/train_tree_baseline.py`
- `scripts/analysis/compute_player_dax_scores.py`

**See:** `docs/BASELINE_MODELING_STRATEGY.md` for complete strategy

