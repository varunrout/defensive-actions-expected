# Validation methodology

Models predict opponent threat at the defensive-action timestamp. Primary variants use `pre_action_context`; post-action fields are diagnostic-only and prohibited from primary contracts.

## The canonical match-grouped split

Validation is grouped by `match_id` (never row-level) because a possession-autocorrelation check found `target_future_shot_10s` heavily clustered within possessions (ICC 0.27-0.49 depending on leg) but barely differing between matches (ICC 0.006-0.008) -- match-level grouping costs almost nothing in signal and is the only grouping that fully rules out possession/event leakage between train and validation.

**As of this pass, there is a real, frozen split.** Before it, the only fold-membership artifacts present in `outputs/models/splits/` were per-baseline outputs generated against a 4-match unit-test fixture (`match_id` `"1"`-`"4"`) -- not a real run over the 115-match dataset. (A handful of `*_with_360`/`*_360_geometry` baseline runs had already used real match ids via `make_grouped_folds()` directly, but each drew its own independent random k-fold split with no held-out test set and no sharing between the active and passive legs -- still not the canonical, comparable split described below.)

The canonical split is computed once by `scripts/pipeline/compute_canonical_split.py` and frozen at `outputs/models/splits/match_assignment.json` (`{match_id: "test" | "fold0".."fold4"}`, all 115 matches, checked into git as a versioned artifact -- see the `.gitignore` exception carved out for this one file). It is built as:

1. Per match, per leg (active and passive), compute the `target_future_shot_10s` rate. Combine as the average of each leg's own percentile rank into one `combined_rank` per match (not either leg's raw rate alone -- splitting each leg independently was tried first and rejected, since it put only 5 of 23 test matches in common between the two legs, ruling out ever comparing active-model vs. passive-model performance on identical held-out matches).
2. `train_test_split(matches, test_size=0.20, stratify=qcut(combined_rank, 4), random_state=42)` -> 23 held-out TEST matches, 92 TRAIN+VAL matches.
3. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on the 92-match TRAIN+VAL pool, grouped by `match_id`, stratified on the row-level `target_future_shot_10s` (computed on the active leg's rows; grouping is match-level, so this does not change which matches land in which fold).

Both legs read the same `match_id -> split label` mapping via `dax.models.splits.load_canonical_split()`, so a match held out as TEST (or assigned to a given fold) is the same match for both legs. `dax.models.splits.canonical_grouped_folds()` and `canonical_test_mask()` are the row-level helpers built on top of it; `train_variant()` in `training.py` excludes canonical-TEST-match rows before fold-making, and `make_valid_variant_folds()` prefers the canonical split whenever a dataset's matches are fully covered by it, falling back to the legacy `make_grouped_folds()` retry-fewer-folds logic otherwise (unit-test fixtures, or any dataset built after the split was frozen).

`make_grouped_folds()` itself is unchanged and still available for exploratory/ad-hoc use -- the canonical split is the default path for real model runs, not a replacement for the underlying grouped-CV mechanism.

Three integrity properties are enforced as regression tests (`tests/test_canonical_split.py`), not just checked once manually: no match appears in two CV folds, no TEST match appears in any CV fold, and every one of the 115 matches is accounted for exactly once.

## General grouped-fold mechanics

For role-dependent 360 variants, rows are filtered before folds are created. The exact default rule is: `has_360` is true, `freeze_frame_roles_known` is true when `require_roles_known` is set, local visibility flags are true when explicitly required by the contract, and every required role-dependent 360 feature is non-null. Visibility distribution uses the canonical `visibility_quality_band` field.

For every fold, the framework records rows, matches, positive-shot support, non-zero future-xG support, target mean/prevalence, calibration method/support where relevant, task metrics, fit time, and inference time. Aggregate model comparison rows include full-OOF metrics plus fold mean, standard deviation, minimum, and maximum summaries.

Because 360 variants can use a different eligible population, the framework writes labelled native-population, common-row diagnostic, all-data non-360, and 360-only comparison tables. Common-row tables are explicitly diagnostic until row-restricted rescoring is run on real data.
