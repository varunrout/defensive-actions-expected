# Baseline Model Results (Initial Smoke Run)

Date: 2026-06-10

This is the first validated baseline modeling run using a 10,000-row subset to verify the end-to-end training/evaluation flow.

## Setup

- Script: `scripts/models/train_baseline_logistic.py`
- Script: `scripts/models/evaluate_baseline_model.py`
- CV strategy: GroupKFold by `match_id` (5 folds)
- Target: `target_shot_in_10s`

## Smoke Metrics (10,000 rows)

| Variant | ROC-AUC | Avg Precision |
|---|---:|---:|
| `v0_phase_only` | 0.6615 | 0.1159 |
| `v1_spatial` | 0.7622 | 0.2615 |
| `v2_full_baseline` | 0.7807 | 0.2745 |

## Artifacts Generated

- `outputs/models/baseline/logistic_v0_phase_only.joblib`
- `outputs/models/baseline/logistic_v1_spatial.joblib`
- `outputs/models/baseline/logistic_v2_full_baseline.joblib`
- `outputs/validation/baseline/baseline_model_metrics.json`
- `outputs/oof/baseline/baseline_oof_predictions.parquet`
- `outputs/validation/baseline/baseline_roc_curves.png`
- `outputs/validation/baseline/baseline_pr_curves.png`
- `outputs/validation/baseline/baseline_feature_importance.png`

## Next Step

Run full-data training (without `--max-rows`) and compare stability of fold metrics by phase and match groups.
