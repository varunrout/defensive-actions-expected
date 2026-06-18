# MLflow tracking

MLflow is an optional modelling dependency. The modelling environment should be installed with:

```bash
python -m pip install -e ".[dev,visualization,models]"
```

`configs/models.yaml` enables MLflow for modelling runs by default. Use `--no-mlflow-enabled` for local dry runs, legacy smoke tests, or environments where the `models` extra has not been installed. If MLflow is enabled but unavailable, the framework raises a clear error containing the installation command above.

Tracking URI precedence is:

1. explicit CLI `--tracking-uri`
2. `MLFLOW_TRACKING_URI`
3. `mlflow.tracking_uri` in the configuration
4. MLflow's local default

The code does not silently fall back to another destination when an explicit remote URI is supplied. Remote HTTP(S) stores are checked before training starts; unreachable stores raise an error.

For a local UI, optionally run:

```bash
mlflow server --port 5000
```

A server is not required for local file tracking.
