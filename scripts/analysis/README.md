# Analysis Scripts

Statistical and machine learning analysis of defensive features.

## Scripts

- **run_player_feature_analysis.py** - Rigorous statistical validation pipeline
  - Runs 10 modular validation modules
  - Modules: schema quality, target audit, univariate signal, interactions, stratification, leakage, stability, redundancy, negative controls, decision report
  - Outputs: detailed analysis reports, charts, summary JSON
  - Decision: GO_BASELINE (passes all gates) vs HOLD (needs remediation)

- **feature_stability_report.py** - Coefficient and correlation stability diagnostics
  - Outputs correlation clusters for numeric features
  - Computes fold-by-fold coefficient sign stability (GroupKFold)
  - Runs tactical-vs-proxy ablation table
  - Produces recommended interpretable feature set CSV

- **feature_selection_clustering_pca.py** - Feature pruning analysis for next variants
  - Builds numeric correlation clusters and cluster representatives
  - Runs PCA diagnostics (explained variance + top loadings)
  - Produces v5 (interpretable) and v6 (balanced) selected-feature manifests

- **build_modeling_report.py** - Portfolio comparison and report generation
  - Aggregates OOF metrics across all trained logistic and regression variants
  - Computes cross-task alignment and fold-stability summaries
  - Reads slice-analysis, feature-stability, and clustering artifacts
  - Generates a full markdown report in `docs/analysis/`

- **build_defensibility_addendum.py** - Extra evidence package for model defensibility
  - Computes match-bootstrap confidence intervals for classification and regression metrics
  - Adds winner-frequency robustness checks under resampling
  - Produces calibration reliability curves and ECE summary for top classifiers
  - Produces regression decile-bias diagnostics and focused `v4-v3` slice gap table
  - Generates `docs/analysis/DEFENSIBILITY_ADDENDUM.md`

## Usage

```bash
python analysis/run_player_feature_analysis.py
```

```bash
python scripts/analysis/feature_stability_report.py --task logistic --variant v4_freeze_geometry
python scripts/analysis/feature_stability_report.py --task regression --variant v3_context_enhanced
```

```bash
python scripts/analysis/feature_selection_clustering_pca.py --task logistic --variant v4_freeze_geometry
python scripts/analysis/feature_selection_clustering_pca.py --task regression --variant v4_freeze_geometry
```

```bash
python scripts/analysis/build_modeling_report.py
```

```bash
python scripts/analysis/build_defensibility_addendum.py --n-boot 250 --seed 42
```

## Output

- `outputs/validation/analysis/player_features/summary.json` - Overall decision + module results
- `outputs/validation/analysis/player_features/tables/` - CSV outputs per module
- `outputs/validation/analysis/player_features/` - Analysis charts and visualizations
- `outputs/validation/analysis/feature_stability/<task>_<variant>/correlation_clusters.csv`
- `outputs/validation/analysis/feature_stability/<task>_<variant>/coefficient_sign_stability.csv`
- `outputs/validation/analysis/feature_stability/<task>_<variant>/tactical_vs_proxy_ablation.csv`
- `outputs/validation/analysis/feature_stability/<task>_<variant>/interpretable_feature_set_recommended.csv`
- `outputs/validation/analysis/feature_stability/<task>_<variant>/summary.json`
- `outputs/validation/analysis/feature_selection/<task>_<variant>/correlation_clusters.csv`
- `outputs/validation/analysis/feature_selection/<task>_<variant>/feature_selection_decisions.csv`
- `outputs/validation/analysis/feature_selection/<task>_<variant>/selected_features_v5.csv`
- `outputs/validation/analysis/feature_selection/<task>_<variant>/selected_features_v6.csv`
- `outputs/validation/analysis/feature_selection/<task>_<variant>/feature_selection_manifest.json`
- `outputs/validation/analysis/feature_selection/<task>_<variant>/correlation_heatmap_clustered.png`
- `outputs/validation/analysis/feature_selection/<task>_<variant>/pca_explained_variance.png`
- `outputs/validation/analysis/feature_selection/<task>_<variant>/pca_top_loadings.png`
- `outputs/validation/comparison/model_portfolio/model_leaderboard_all.csv`
- `outputs/validation/comparison/model_portfolio/cross_task_alignment.csv`
- `outputs/validation/comparison/model_portfolio/fold_stability_summary.csv`
- `outputs/validation/comparison/model_portfolio/slice_win_counts.csv`
- `outputs/validation/comparison/model_portfolio/model_portfolio_summary.json`
- `docs/analysis/MODEL_PORTFOLIO_REPORT.md`
- `outputs/validation/comparison/defensibility/classification_metric_ci.csv`
- `outputs/validation/comparison/defensibility/classification_winner_frequency.csv`
- `outputs/validation/comparison/defensibility/regression_metric_ci.csv`
- `outputs/validation/comparison/defensibility/regression_winner_frequency.csv`
- `outputs/validation/comparison/defensibility/classification_calibration_summary.csv`
- `outputs/validation/comparison/defensibility/regression_decile_bias_summary.csv`
- `outputs/validation/comparison/defensibility/slice_gap_v4_vs_v3.csv`
- `outputs/validation/comparison/defensibility/calibration_reliability_curves.png`
- `outputs/validation/comparison/defensibility/regression_decile_bias.png`
- `outputs/validation/comparison/defensibility/defensibility_summary.json`
- `docs/analysis/DEFENSIBILITY_ADDENDUM.md`

## Results

Current run (unidirectional normalized):
- **Decision**: GO_BASELINE ✓
- **Rows analyzed**: 57,637 defensive actions
- **Shot-in-10s rate**: 7.95%
- **Players**: 1,012 unique
- **Matches**: 115 tournaments
