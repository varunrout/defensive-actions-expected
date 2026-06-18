# Modelling workflow

This modelling phase is leakage-safe and provisional. It uses `data/features/player_defensive_actions.parquet`, grouped validation by `match_id`, targets `target_future_shot_10s` and `target_future_xg_10s`, explicit contracts in `configs/models.yaml`, and MLflow tracking when enabled.

Install modelling dependencies with:

```bash
python -m pip install -e ".[dev,visualization,models]"
```

The default local MLflow tracking backend is SQLite:

```text
sqlite:///mlflow.db
```

To inspect local runs in a UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Remote tracking can be configured with `--tracking-uri`, `MLFLOW_TRACKING_URI`, or the `mlflow.tracking_uri` configuration value. If a remote URI is explicitly supplied, failures should be surfaced rather than silently redirected.

Player signals are out-of-fold expected-versus-observed summaries only. They are not true DAx and are not causal estimates.
