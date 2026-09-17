# Active-Binary Baseline Summary (v0-v3)

Target: `target_future_shot_10s`. Dataset: `data/features/player_defensive_actions.parquet`
(56,068 rows / 115 matches -- confirmed real scale, not the 20-row/4-match
fixture scale that the retired pre-lock `b0`-`b8` classification outputs used).

## 1. Feature set

32 of the 34 locked ACTIVE features from `src/eda/feature_config.py` (`ACTIVE`
dict: 6 categorical + 9 boolean + 12 continuous + 7 discrete = 34), excluding:

- `nearest_defender_distance` -- known self-reference bug (~73.6% of rows
  approx 0m per the distribution atlas), not a genuine defensive-distance signal.
- `defenders_within_5m` -- confirmed substitutive/nested with
  `defenders_within_10m` for this target (`reports/eda/FEATURE_INTERACTION_ANALYSIS.json`:
  "defenders_within_5m x defenders_within_10m -> substitutive").

Resulting count verified programmatically at training time (`scripts/train_active_binary_baseline.py`
asserts `len(ALL_FEATURE_COLS) == 32`): 6 categorical (one-hot), 9 boolean
(passthrough), 11 continuous + 6 discrete = 17 numeric (median-impute +
standard-scale).

None of the 5 pre-lock-dropped columns (`action_x`, `action_y`, `action_zone`,
`action_family`, `position_group`, `distance_to_center_line`) appear anywhere
in this feature set.

## 2. Split

The canonical, frozen match-grouped split in
`outputs/models/splits/match_assignment.json` (5-fold, seed 42, computed once
by `scripts/compute_canonical_split.py`). No fresh random split was created
anywhere in this run. Train+val: 45,408 rows / 92 matches. Held-out test:
10,660 rows / 23 matches.

## 3. Variants

| Variant | Model | Notes |
|---|---|---|
| `v0_dummy` | `ConstantClassifier` | training-fold positive rate for every row |
| `v1_unweighted` | `LogisticRegression` | no class_weight |
| `v2_weighted` | `LogisticRegression(class_weight="balanced")` | same features as v1 |
| `v3_weighted_interactions` | v2 + 11 explicit interaction terms | 4 numeric x numeric pairs + 7 `phase_label` dummy terms for `defenders_within_10m x phase_label` |

The 5 interaction pairs were cross-checked against
`reports/eda/FEATURE_INTERACTION_ANALYSIS.json` and `reports/eda/MASTER_FINDINGS.md`
section 6 before use. The 4 numeric x numeric pairs are each independently
confirmed `"interactive"` for `target_future_shot_10s` in
`FEATURE_INTERACTION_ANALYSIS.json`'s `summary_table`. The 5th pair
(`defenders_within_10m x phase_label`) is **not** in that JSON (it only covers
numeric x numeric pairs) -- it comes from `MASTER_FINDINGS.md` section 6
("Active-binary"), which states it "diverges in 7/7 phase categories -- the
single strongest slice-level interaction signal in the whole corpus." This
matches the task brief exactly.

## 4. Cross-validation results (5-fold, out-of-fold metrics)

| Variant | PR-AUC (AP) | ROC-AUC | Log loss | Brier | Calib. slope | Calib. intercept | ECE |
|---|---|---|---|---|---|---|---|
| v0_dummy | 0.0751 | 0.4747 | 0.2772 | 0.0730 | 0.857 | -0.350 | 0.0000 |
| **v1_unweighted** | **0.3521** | 0.8177 | **0.2209** | **0.0614** | 0.972 | **-0.052** | **0.0036** |
| v2_weighted | 0.3376 | **0.8188** | 0.5151 | 0.1739 | 0.990 | -2.432 | 0.2855 |
| v3_weighted_interactions | 0.3331 | 0.8186 | 0.5153 | 0.1739 | 0.980 | -2.430 | 0.2850 |

Per-fold mean +/- std (average precision, the primary metric): v0 0.0792
+/- 0.0069; v1 0.3531 +/- 0.0176; v2 0.3389 +/- 0.0163; v3 0.3349 +/- 0.0148.
Full per-metric fold mean/std for every variant is in
`outputs/models/comparisons/active_binary_baseline_comparison.csv`.

**PR-AUC ranking: v1 > v2 > v3 > v0.** The plain unweighted logistic
regression (v1) has the *best* PR-AUC of the three fitted variants in
cross-validation, ahead of both weighted variants. `class_weight="balanced"`
(v2/v3) buys a very small ROC-AUC gain (+0.001) at a large calibration cost:
calibration intercept collapses from -0.05 (v1) to -2.43 (v2/v3) and expected
calibration error jumps from 0.004 to 0.285 -- balanced weighting inflates
predicted probabilities well past the true positive rate, so log loss and
Brier score both roughly triple. This is the expected mechanical effect of
`class_weight="balanced"` on a linear classifier's raw probabilities, not a
sign of a broken model, but it means v2/v3's `predict_proba` output should
not be read as a calibrated probability without a separate calibration step
(out of scope here).

**Does v3's interaction terms beat v2 by enough to matter? No.** v3's PR-AUC
(0.3331 CV / 0.3572 held-out) is slightly *below* v2's (0.3376 CV / 0.3594
held-out) in both the cross-validation and the held-out readout. ROC-AUC is
statistically indistinguishable (0.8186 vs 0.8188 CV). The 11 interaction
coefficients are mostly small relative to the main-effect coefficients (see
`outputs/models/classification/v3_weighted_interactions.json`); the two
largest are the `defenders_within_10m x phase_label_settled_mid_block_proxy`
(+0.191) and `x phase_label_wide_defending_proxy` (-0.171) terms, consistent
with the documented "strongest slice-level interaction" finding in direction
of effect, but the net effect on this held-out linear model's discrimination
and ranking is a wash to a small regression, not an improvement. This does
not contradict the underlying interaction finding (which was established via
direct stratified rate analysis, not via a linear model's coefficient) --
it means a simple product term added to an already class-weighted linear
model does not translate that finding into a better-ranking baseline here.

## 5. Held-out test readout (v1/v2/v3, single readout, NOT used for selection)

| Variant | PR-AUC | ROC-AUC | Log loss | Brier | Calib. slope | Calib. intercept | ECE |
|---|---|---|---|---|---|---|---|
| v1_unweighted | 0.3725 | 0.8193 | 0.2047 | 0.0555 | 1.002 | -0.122 | 0.0107 |
| v2_weighted | 0.3594 | 0.8210 | 0.5193 | 0.1761 | 1.029 | -2.587 | 0.2940 |
| v3_weighted_interactions | 0.3572 | 0.8206 | 0.5189 | 0.1759 | 1.023 | -2.583 | 0.2934 |

10,660 rows / 23 matches, evaluated exactly once each, after all
cross-validation and variant comparison above was finalised. Consistent with
the CV ranking: v1 leads on PR-AUC and by a wide margin on calibration; v2
slightly ahead of v3.

**Model provenance for this readout:** v2/v3 reuse the model objects already
fit on all train+val rows in section 5's original run
(`outputs/models/classification/v2_weighted.joblib` /
`v3_weighted_interactions.joblib`). v1 was refit on the identical train+val
rows, features and settings (`LogisticRegression(random_state=42)`, no
`class_weight`) rather than reloading `v1_unweighted.joblib`, since that
earlier artifact and its `DesignMatrixBuilder` preprocessing state were not
both persisted together in a directly reloadable form. The refit intercept
(-1.31643790**47863153**) matches the original saved fit
(-1.31643790**3595758**) to 8 significant figures -- the residual is
solver-tolerance floating-point noise from `lbfgs`, not a different fit on
different data. The held-out test set was still touched exactly once for
this purpose.

**v1 CV-to-test check (added after prompt 37 -- follow-up to the overfitting
question raised in discussion):** v1's PR-AUC went from 0.3521 (CV mean) to
0.3725 (held-out test) -- an *increase*, not a drop. This is the same
direction v2 (0.3376 CV -> 0.3594 test) and v3 (0.3331 CV -> 0.3572 test)
both showed. Calibration on the held-out set stays excellent (slope 1.002,
intercept -0.122, ECE 0.0107 -- essentially unchanged from CV's ECE of
0.0036, both far better than v2/v3's 0.29). No evidence of overfitting: the
model was not memorising training-fold-specific patterns, it generalises to
the 23 held-out matches at least as well as it performed in cross-validation.
Combined with the earlier evidence (fold std of only +/-0.0176 on PR-AUC
across 5 match-grouped folds, L2 regularization by default, ~825 training
rows per effective parameter), this closes out the generalisation question
for v1 -- the CV result was real, not a fold-specific fluke.

## 6. Coefficient-vs-shape sanity check (v2/v3 vs `active_numerical_target_atlas.json`)

Checked every continuous/discrete feature with a documented shape in
`reports/eda/active_numerical_target_atlas.json` against its sign in
`outputs/models/classification/v2_weighted.json` (v3's coefficients agree in
sign throughout, so only v2 is quoted below).

**Confirms (sign of linear coefficient matches documented monotonic direction), 6 of 11:**

| Feature | Documented shape (rho) | v2 coefficient | Verdict |
|---|---|---|---|
| `attacking_goal_centrality` | monotonic increasing (+0.170) | +0.550 | confirms -- largest continuous coefficient, matches strongest-correlate ranking |
| `attacker_spread` | monotonic decreasing (-0.170) | -0.121 | confirms |
| `attacker_defender_ratio` | monotonic decreasing (-0.156) | -0.269 | confirms |
| `defenders_within_10m` | monotonic increasing (+0.152) | +0.077 | confirms |
| `match_time_seconds` | monotonic increasing (+0.035) | +0.104 | confirms |
| `possession_elapsed_seconds` | monotonic increasing (+0.067) | +0.055 | confirms |

**Mismatch -- linear model cannot represent the documented shape, 4 of 11:**

| Feature | Documented shape (rho) | v2 coefficient | Issue |
|---|---|---|---|
| `defender_spread` | **U-shaped** (rho=-0.188, strongest active correlate) | -0.143 | A single linear coefficient can only report a net monotonic direction. The documented U-shape means risk is elevated at *both* very low and very high defender spread, which this coefficient cannot represent -- it reports "spread down -> risk up" everywhere, which is only half the true (documented) relationship. |
| `distance_to_attacking_box` | U-shaped (rho=-0.019, weak) | +0.112 | Coefficient sign doesn't even match the (weak) overall Spearman direction; both are likely dominated by the non-monotonic part of the true relationship that a linear term cannot see. |
| `visible_defender_count` | U-shaped (rho=+0.087) | +0.253 | Same blind spot as `defender_spread`: net-positive linear coefficient, but the documented shape says risk rises at both extremes, not monotonically. |
| `defenders_between_ball_and_attacking_goal` | U-shaped (rho=+0.013, near-zero) | -0.032 | Sign disagreement on an already very weak univariate signal -- likely just noise on both sides for this feature. |

**Flagged, no clear univariate support:**

`nearest_attacker_distance` carries a moderately large v2 coefficient
(+0.226) despite `active_numerical_target_atlas.json` documenting
"no-clear-pattern" (rho=0.023) for it univariately. This coefficient is most
plausibly picking up a multivariate/interaction effect with correlated
features (e.g. `visible_attacker_count`, `attacker_spread`) rather than a
univariately-real relationship -- worth a VIF/partial-dependence follow-up
before trusting it in isolation.

**Overall**: 6/11 checked features have linear-model coefficients that agree
in sign with their documented monotonic pattern; the other 5 either have a
genuinely non-monotonic (U-shaped) documented pattern that a linear logistic
term structurally cannot capture, or (for `nearest_attacker_distance`) show
a coefficient not well-supported by the univariate atlas evidence. This is
an honest limitation of a linear baseline, not a bug -- a non-linear baseline
(e.g. the existing `hist_gradient_boosting_classifier` variants in
`configs/models.yaml`) would be needed to capture the U-shaped relationships
directly.

## 7. Deviations from instructions

- `configs/models.yaml`'s `classification.variants` (`b0_constant` ...
  `b8_interpretable_reduced`) were **not** modified, even though they are the
  actual source of the retired `outputs/models/classification/b*` artifacts
  (not `src/dax/models/baseline_logistic.py`'s `default_variant_specs()`,
  which the task background attributed them to). Investigation found
  `default_variant_specs()` was dead code exercised only by two test files
  (`tests/test_methodology_invariants.py`, `tests/test_model_spec_schema.py`)
  and an already-archived `src/dax/models/archive/specs.py` -- it never fed
  the real training entrypoint (`scripts/train_models.py` ->
  `dax.models.training.train_logistic_models`, which reads
  `configs/models.yaml` via `feature_contracts.py`). Editing
  `configs/models.yaml` would have been destructive to unrelated, working
  functionality that the task did not ask to touch: `b5_360_geometry` and
  `b7_full_with_360` (real-scale, 56,024 rows/115 matches, not toy-scale) and
  `r4_full_with_360` feed the two-part xG hurdle-model sensitivity analysis
  (per prior commit history), and `tests/test_model_workflow_production_fixture.py`
  asserts against `b5_360_geometry`/`b0_constant` directly. The 20-row/4-match
  toy-scale symptom is real for most `b0`-`b8`/`r0`-`r6` rows, but this is a
  property of the *default* `train_models.py` invocation (small `max_rows`),
  not of `configs/models.yaml` itself -- `b5`/`b7`/`r4` prove the same config
  produces real-scale output when required-360 filtering already narrows the
  input.
- The generated Part B artifacts under `outputs/models/**` (comparison CSVs,
  per-variant JSON/joblib, charts) are **not** git-tracked, matching this
  repository's existing `.gitignore` convention (`/outputs/models/**` is
  ignored except `outputs/models/splits/match_assignment.json`, and the
  retired `b0`-`b8` artifacts were likewise never tracked). They are
  generated on disk by `scripts/train_active_binary_baseline.py` and are
  reproducible from a single command; only code and `reports/eda/*` changes
  are committed.
