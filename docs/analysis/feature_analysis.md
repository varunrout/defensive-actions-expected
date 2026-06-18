# Feature Analysis

## Feature groups

Feature diagnostics group columns into identifiers, targets, spatial features, action context, possession semantics, phase proxies, 360 attacker/defender roles, local numerical balance, visibility, exposure, and player aggregates.

## Diagnostics

Numeric diagnostics include missingness, moments, quantiles, unique counts, zero rates, IQR outliers, constant flags, correlations, and univariate descriptive relationships with future-shot and future-xG targets. Categorical diagnostics include frequency concentration, rare-category rates, and target means where sample sizes permit.

## Missingness policy

Required schema fields are never silently skipped. Optional diagnostics are generated only for available canonical production fields. Configured missingness thresholds are used by clustering preprocessing to remove unsuitable style features.

## Interpretation rules

Relationships are descriptive only and must not be interpreted as causal. Phase labels are rule-based tactical proxies. Future targets are observed post-action outcomes, not player value estimates.
