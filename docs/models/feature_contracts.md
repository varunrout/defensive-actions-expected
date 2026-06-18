# Feature contracts

Model variants are defined in `configs/models.yaml`. Each variant declares categorical features, numeric features, required features, optional features, excluded features, target, model family, preprocessing, hyperparameters, minimum usable rows, feature scope, and whether 360 data is required.

360 contracts may also declare:

- `require_roles_known`
- `require_reliable_5m_visibility`
- `require_reliable_10m_visibility`

A 360-specific variant is eligible only after its row-level eligibility rule is applied before fold construction. Missing required features fail the run. Missing optional features are reported and do not silently convert a full model into a heavily reduced model without an audit artifact.

For each variant, training records requested features, available features, missing required features, missing optional features, final fitted features, feature missingness, rows retained/excluded, matches retained/excluded, 360 coverage, and visibility-quality distribution where applicable.
