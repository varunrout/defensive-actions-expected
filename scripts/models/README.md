# Baseline Modeling Scripts

This folder starts Phase 2 baseline modeling for DAx.

## Scripts

- `train_baseline_logistic.py`
  - Trains logistic variants: `v0_phase_only`, `v1_spatial`, `v2_full_baseline`, `v3_context_enhanced`, `v4_freeze_geometry`, `v5_interpretable_clustered`, `v6_balanced_clustered`, `v7_interpretable_ridge`, `v8_balanced_ridge`
  - Uses GroupKFold by `match_id`
  - Saves models to `outputs/models/baseline/`
  - Saves metrics/tables to `outputs/validation/baseline/`
  - Saves OOF predictions to `outputs/oof/baseline/`

- `evaluate_baseline_model.py`
  - Reads OOF predictions and summary metrics
  - Saves ROC and PR plots
  - Saves top coefficient charts (V2-V8)

- `train_baseline_regression.py`
  - Trains regression variants: `v0_phase_only`, `v1_spatial`, `v2_full_baseline`, `v3_context_enhanced`, `v4_freeze_geometry`, `v5_interpretable_clustered`, `v6_balanced_clustered`, `v7_interpretable_ridge`, `v8_balanced_ridge`
  - `v5` and `v6` use `LinearRegression`; `v7` and `v8` apply `Ridge` regularization on the same clustered feature templates
  - Uses GroupKFold by `match_id`
  - Saves models to `outputs/models/regression/`
  - Saves metrics/tables to `outputs/validation/regression/`
  - Saves OOF predictions to `outputs/oof/regression/`

- `evaluate_baseline_regression.py`
  - Reads regression OOF predictions and summary metrics
  - Saves regression metric comparison chart
  - Saves predicted-vs-actual and residual charts
  - Saves top coefficient charts (V2-V8)

- `run_baseline_modeling.ps1`
  - Convenience wrapper to run train + evaluate

- `run_baseline_regression.ps1`
  - Convenience wrapper to run regression train + regression charts + optional classification-vs-regression comparison

## Outputs

- `outputs/models/baseline/logistic_*.joblib`
- `outputs/validation/baseline/baseline_model_metrics.json`
- `outputs/validation/baseline/baseline_model_metrics_table.csv`
- `outputs/oof/baseline/baseline_oof_predictions.parquet`
- `outputs/validation/baseline/baseline_roc_curves.png`
- `outputs/validation/baseline/baseline_pr_curves.png`
- `outputs/validation/baseline/baseline_v*_feature_importance.png`
- `outputs/models/regression/regression_*.joblib`
- `outputs/validation/regression/regression_model_metrics.json`
- `outputs/validation/regression/regression_model_metrics_table.csv`
- `outputs/oof/regression/regression_oof_predictions.parquet`
- `outputs/validation/regression/regression_metrics_by_variant.png`
- `outputs/validation/regression/regression_predicted_vs_actual.png`
- `outputs/validation/regression/regression_residuals.png`
- `outputs/validation/regression/regression_v*_feature_importance.png`

## Quick Run

```powershell
.\.venv\Scripts\python.exe scripts\models\train_baseline_logistic.py --max-rows 20000
.\.venv\Scripts\python.exe scripts\models\evaluate_baseline_model.py
```

Or wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\models\run_baseline_modeling.ps1 -MaxRows 20000
```

Regression run:

```powershell
.\.venv\Scripts\python.exe scripts\models\train_baseline_regression.py --max-rows 20000
.\.venv\Scripts\python.exe scripts\models\evaluate_baseline_regression.py
```

Regression workflow wrapper (recommended after baseline):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\models\run_baseline_regression.ps1 -MaxRows 20000
powershell -ExecutionPolicy Bypass -File scripts\models\run_baseline_regression.ps1 -SkipTrain
powershell -ExecutionPolicy Bypass -File scripts\models\run_baseline_regression.ps1 -Variant v2_full_baseline
```
