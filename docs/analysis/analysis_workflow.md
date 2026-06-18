# Analysis Workflow

Run pre-modelling analysis after data preparation and player feature construction, and before final predictive model training:

```text
prepare data → build features → analyse processed data → analyse features → build player summary → run clustering → build descriptive signals → generate analysis report → assess model readiness → train models later
```

Canonical commands are documented in the README. The six active analysis CLIs import reusable logic from `src/dax/analysis/` and write outputs below `outputs/analysis/` plus generated player feature tables below `data/features/`.
