# Pipeline architecture

The active pipeline is represented by five script entry points and reusable package modules under `src/dax/`.

1. `scripts/run_pipeline.py` declares the overall stage sequence.
2. `scripts/build_features.py` is the feature-build entry point.
3. `scripts/train_models.py` is the model-training entry point.
4. `scripts/validate_models.py` is the validation entry point.
5. `scripts/generate_reports.py` is the reporting entry point.

Reusable calculations remain in `src/dax/`. The active scripts are intentionally thin and contain no `sys.path.insert` calls.
