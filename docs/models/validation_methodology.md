# Validation methodology

Models predict opponent threat at the defensive-action timestamp. Primary variants use `pre_action_context`; post-action fields are diagnostic-only and prohibited from primary contracts.

Validation uses grouped folds by `match_id`. Each variant receives fold assignments over its eligible row population. If requested folds produce unsupported validation folds, the splitter retries with fewer grouped folds, records the fallback sequence, saves the effective fold count, and fails only when no valid grouped split exists. Classification validation folds must contain positive and negative targets; regression validation folds must contain non-zero future-xG support.

For role-dependent 360 variants, rows are filtered before folds are created. The exact default rule is: `has_360` is true, `freeze_frame_roles_known` is true when `require_roles_known` is set, local visibility flags are true when explicitly required by the contract, and every required role-dependent 360 feature is non-null. Visibility distribution uses the canonical `visibility_quality_band` field.

For every fold, the framework records rows, matches, positive-shot support, non-zero future-xG support, target mean/prevalence, calibration method/support where relevant, task metrics, fit time, and inference time. Aggregate model comparison rows include full-OOF metrics plus fold mean, standard deviation, minimum, and maximum summaries.

Because 360 variants can use a different eligible population, the framework writes labelled native-population, common-row diagnostic, all-data non-360, and 360-only comparison tables. Common-row tables are explicitly diagnostic until row-restricted rescoring is run on real data.
