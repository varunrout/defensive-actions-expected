# Validation methodology

Models predict opponent threat at the defensive-action timestamp. Primary variants use `pre_action_context`; post-action fields are diagnostic-only and prohibited from primary contracts.

Validation uses grouped folds by `match_id`. Each variant receives fold assignments over its eligible row population. For 360-specific variants, rows are filtered before folds are created using the variant eligibility rule. Fold metadata is saved separately from row-level OOF predictions so train-match lists are not repeated per row.

For every fold, the framework records rows, matches, positive-shot support, non-zero future-xG support, target mean/prevalence, task metrics, fit time, and inference time. Aggregate model comparison rows include full-OOF metrics plus fold mean, standard deviation, minimum, and maximum summaries.

Impossible splits fail clearly: classification validation folds must contain positive and negative targets, and regression validation folds must report non-zero future-xG support.
