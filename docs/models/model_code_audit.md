# Model code audit

Prediction timestamp: the primary modelling question predicts opponent threat following a defensive action using information available at the action timestamp. Action outcome fields are excluded from primary `pre_action_context` models; `post_action_observed` is reserved for clearly labelled diagnostics.

| File | Current purpose | Input schema | Target | Model type | Features | CV | Artifacts | Validity / duplication / stale columns / leakage | Decision |
|---|---|---|---|---|---|---|---|---|---|
| `scripts/train_models.py` | CLI wrapper for the new framework | production feature parquet | shot/xG 10s | configured variants | `configs/models.yaml` contracts | grouped by `match_id` | models, OOF, comparison tables, MLflow | valid active entry point | REWRITE |
| `scripts/validate_models.py` | Validate saved OOF outputs | OOF parquet | shot/xG 10s | validation only | n/a | fold isolation checks | validation summary | valid | REWRITE |
| `scripts/generate_reports.py` | Legacy general report script | mixed analysis outputs | n/a | n/a | n/a | n/a | static report | not modelling-specific | KEEP |
| `scripts/models/train_baseline_logistic.py` | Legacy baseline script | feature parquet | `target_future_shot_10s` | logistic | V0-V8 loose columns | grouped | joblib/csv | duplicated by new package; historical compatibility only | ARCHIVE LATER |
| `scripts/models/train_baseline_regression.py` | Legacy regression script | feature parquet | `target_future_xg_10s` | linear | loose columns | grouped | joblib/csv | duplicated by new package; historical compatibility only | ARCHIVE LATER |
| `src/dax/models/baseline_logistic.py` | legacy utilities | production-ish | shot 10s | logistic | stale V variants | grouped | metrics | stale loose resolution can silently drop features | MERGE/REWRITE |
| `src/dax/models/baseline_regression.py` | legacy utilities | production-ish | xG 10s | ridge/elastic net | stale V variants | grouped | metrics | stale loose resolution can silently drop features | MERGE/REWRITE |
| `src/dax/models/training.py` | new orchestrator | strict production schema | shot/xG 10s | baselines/linear/HGB | strict contracts | grouped by match | local + MLflow | valid | KEEP |
| `configs/model_spec.yaml` | older spec if present | n/a | n/a | n/a | loose spec | n/a | n/a | replaced by `configs/models.yaml` | ARCHIVE/DELETE if present |
| `configs/models.yaml` | new canonical model config | production schema | shot/xG 10s | explicit variants | required/optional/excluded | grouped | contract snapshots | valid | KEEP |

Active code must not import from archive directories. The new framework does not import archived scripts.
