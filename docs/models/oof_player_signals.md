# Modelling documentation

This modelling phase is leakage-safe and provisional. It uses `data/features/player_defensive_actions.parquet`, grouped validation by `match_id`, targets `target_future_shot_10s` and `target_future_xg_10s`, explicit contracts in `configs/models.yaml`, and MLflow tracking when enabled.

Local MLflow file tracking works without a server. To inspect runs in a UI, optionally run:

```bash
mlflow server --port 5000
```

Remote tracking can be configured with `MLFLOW_TRACKING_URI` or the `mlflow.tracking_uri` configuration value. If a remote URI is explicitly supplied, failures should be surfaced rather than silently redirected.

Player signals are out-of-fold expected-versus-observed summaries only. They are not true DAx and are not causal estimates.
