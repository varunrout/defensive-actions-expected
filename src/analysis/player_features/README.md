# Player Feature Analysis Module

This package runs a rigorous pre-model analysis for `player_defensive_actions.parquet` and produces a go/no-go decision for baseline modeling.

## Goals

- verify data quality and schema reliability
- validate target behavior and subgroup uncertainty
- screen feature signal and interaction effects
- stress-test stability, drift, leakage, and negative controls
- output reproducible feature shortlists and decision gates

## Package layout

- `config.py` - runtime thresholds and directories
- `io.py` - canonical load + schema normalization
- `utils_stats.py` - Wilson CI, effect sizes, PSI helpers
- `utils_plots.py` - shared figure helpers
- `m01_* ... m10_*` - stepwise analysis scripts
- `run_all.py` - orchestration entrypoint
- `pipeline.py` - compatibility wrapper
- `__main__.py` - CLI entrypoint

## Analysis steps

1. schema and quality checks
2. target/label audit with confidence intervals
3. univariate numeric/categorical signal checks
4. interaction checks (phase/role contexts)
5. stratified consistency checks
6. leakage and confounding checks
7. grouped stability and drift checks
8. redundancy and feature shortlist selection
9. negative-control sanity tests
10. decision report (`GO_BASELINE` or `HOLD`)

## Run

From repository root:

```powershell
.\.venv\Scripts\python.exe -m analysis.player_features --repo-root .
```

Or with script runner:

```powershell
.\.venv\Scripts\python.exe scripts\run_player_feature_analysis.py
```

## Outputs

- `outputs/validation/analysis/player_features/summary.json`
- `outputs/validation/analysis/player_features/tables/*.csv`
- `outputs/validation/analysis/player_features/figures/*.png`
- `outputs/validation/analysis/player_features/report/player_feature_analysis_report.md`
- `outputs/validation/analysis/player_features/report/decision.json`
- `outputs/validation/analysis/player_features/report/features_core.json`
- `outputs/validation/analysis/player_features/report/features_extended.json`

Figure outputs include:

- CI bars for phase/action/role target rates
- interaction and stratified-correlation heatmaps
- drift and bootstrap-stability charts
- negative-control chart
- pitch maps for defensive action density and location-based shot risk
