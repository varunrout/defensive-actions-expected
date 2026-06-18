# MLflow tracking

MLflow is an optional modelling dependency. The modelling environment should be installed with:

```bash
python -m pip install -e ".[dev,visualization,models]"
```

`configs/models.yaml` enables MLflow for modelling runs by default and uses SQLite for the recommended local tracking backend:

```text
sqlite:///mlflow.db
```

Use `--no-mlflow-enabled` for local dry runs, legacy smoke tests, or environments where the `models` extra has not been installed. If MLflow is enabled but unavailable, the framework raises a clear error containing the installation command above.

Tracking URI precedence is:

1. explicit CLI `--tracking-uri`
2. `MLFLOW_TRACKING_URI`
3. `mlflow.tracking_uri` in the configuration
4. MLflow's default

For local tracking through the environment, use:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

The code does not silently fall back to another destination when an explicit remote URI is supplied. Remote HTTP(S) stores are checked before training starts; unreachable stores raise an error.

For a local UI, run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

`MLFLOW_ALLOW_FILE_STORE=true` may be used only as a legacy compatibility escape hatch for old file-store experiments. SQLite is the recommended local backend.
