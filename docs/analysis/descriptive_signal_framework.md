# Analysis documentation

The pre-modelling analysis layer runs after data preparation and feature building, before final predictive model training. Reusable logic lives in `src/dax/analysis/`; scripts under `scripts/` are thin entry points.

Execution order:

```text
prepare data → build features → analyse processed data → analyse features → build player summary → run clustering → build descriptive signals → generate analysis report → assess model readiness → train models later
```

Outputs are written below `outputs/analysis/` and generated feature tables are written below `data/features/`. Phase labels are rule-based tactical proxies, not ground-truth tactical labels. Descriptive signals are provisional and must not be called true DAx.
