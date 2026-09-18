# Active-Binary Baseline Summary (v0-v3)

> **Status update (Prompt 43):** `v1_unweighted` (this document's pick) has been superseded as
> the active-binary leg's standing reference model by `v1c_systematic_interactions` — see the
> promotion decision in `reports/eda/ACTIVE_BINARY_MODELLING_CLOSEOUT.md` section 7. This
> document remains the accurate historical record of the Rung 0 comparison and is not rewritten.

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

## 8. Post-lock validation (v1_unweighted)

Run by `scripts/validate_active_binary_baselines.py` (single command, no split
recomputation, no hyperparameter changes, existing comparison/held-out-readout
CSVs untouched). Full outputs under `outputs/models/validation/`. One
deviation from the brief: `defender_functional_role` does not exist on
`data/features/player_defensive_actions.parquet` (the ACTIVE dataset these
baselines train on) -- it is a PASSIVE-leg-only column
(`src/dax/features/passive_defense.py`, `PASSIVE["categorical"]` in
`feature_config.py`). Item 4 below uses `position` instead, the closest
per-player categorical actually present in `ACTIVE["categorical"]`.

### 8.1 Is the v1 pick statistically justified?

Paired over the same 5 canonical CV folds (`significance_v1_vs_v2_v3.json`),
refitting all three variants fold-by-fold:

| Comparison | Mean &Delta; PR-AUC | Std &Delta; | Paired t (p) | Wilcoxon (p) |
|---|---|---|---|---|
| v1 vs v2 | +0.0142 | 0.0031 | t=10.26, **p=0.0005** | W=0, p=0.0625 |
| v1 vs v3 | +0.0182 | 0.0070 | t=5.81, **p=0.0044** | W=0, p=0.0625 |

v1 beats both v2 and v3 in **5 of 5 folds** on PR-AUC. The paired t-test says
the gap is statistically significant at the conventional 0.05 threshold for
both comparisons. The Wilcoxon signed-rank test lands at **0.0625 for both**
-- stated plainly, this is *not* below 0.05, but 0.0625 is also the exact
minimum p-value Wilcoxon can produce with only 5 paired samples where every
pair has the same sign (there is no smaller rank-sum configuration to reach
with n=5). So the non-parametric test gives the strongest signal it is
mathematically capable of giving at this sample size, without crossing the
conventional threshold on its own. Read together: the t-test result is
significant, the Wilcoxon result is consistent with but not independently
significant of that conclusion, and the practical signal (5/5 folds, every
fold same direction, mean gap 3-5x the per-fold std) supports treating v1's
CV edge as real rather than fold noise -- with the honest caveat that 5 folds
is a small sample for any significance test and this is not overwhelming
statistical proof.

### 8.2 Does v1 hold up across tournaments?

`tournament_stratified_v1.json`. First finding: **only 2 tournaments are
actually represented in the active dataset**, not 3. UEFA Euro 2020
(`data/raw/matches/55_43.json`, 51 matches) contributes **zero** rows to
`player_defensive_actions.parquet` -- confirmed by checking its 51 match IDs
against the dataset's `match_id` column directly (0 overlap). The dataset's
115 matches are FIFA World Cup 2022 (64) and UEFA Euro 2024 (51) only. This
is a property of the upstream feature dataset, not a bug in this validation
script.

| Tournament | Held-out rows/matches | PR-AUC | ROC-AUC | Log loss | Brier | ECE |
|---|---|---|---|---|---|---|
| FIFA World Cup 2022 | 6,711 / 14 | 0.3596 | 0.8334 | 0.1872 | 0.0506 | 0.0106 |
| UEFA Euro 2024 | 3,949 / 9 | 0.3913 | 0.7971 | 0.2343 | 0.0638 | 0.0158 |

Train+val composition: 50 WC2022 matches / 42 Euro2024 matches. v1 performs
comparably on both held-out slices -- PR-AUC is actually slightly *higher* on
the smaller Euro 2024 slice (0.391 vs 0.360), ROC-AUC slightly lower (0.797 vs
0.833), and calibration stays tight on both (ECE 0.011 / 0.016). No sign of
the model being tournament-specific or overfit to World Cup patterns. (Both
slices clear the >=2-matches/>=2-positives bar for a stable PR-AUC readout;
neither is flagged as too-thin.)

### 8.3 Does v1 survive a real player-disjoint split?

`player_disjoint_v1.json`. 964 unique players in train+val; 5-fold
`GroupKFold` on `player_id` (test-set matches excluded throughout --
the frozen held-out test set was not touched by this check).
`player_overlap_train_test == 0` confirmed and asserted for every fold
(193/193/193/192/193 held-out players per fold).

| | PR-AUC | ROC-AUC | Log loss | Brier | ECE |
|---|---|---|---|---|---|
| Match-grouped CV (original) | 0.3521 | 0.8177 | 0.2209 | 0.0614 | 0.0036 |
| Player-disjoint CV (this check) | 0.3586 &plusmn; 0.0243 | 0.8202 &plusmn; 0.0075 | 0.2198 &plusmn; 0.0088 | 0.0611 &plusmn; 0.0031 | 0.0055 &plusmn; 0.0013 |

The player-disjoint PR-AUC (0.359) is **not meaningfully different** from the
match-grouped CV PR-AUC (0.352) -- a +0.007 difference, well inside the
player-disjoint run's own fold std (&plusmn;0.024) and smaller than the
match-grouped run's own fold std (&plusmn;0.018). Every other metric (ROC-AUC,
log loss, Brier, ECE) is likewise within noise of the original. **This
supports, and does not overturn, the Prompt 31 leakage-clearance finding**:
training on a subset of players and testing on entirely unseen players costs
v1 nothing measurable, which is what "no player-identity leakage" predicts.
Prompt 31's own check only confirmed fold *balance*; this is the first time
v1 has actually been fit and scored on a genuinely player-disjoint split, and
it holds.

### 8.4 Where does v1 systematically miss?

`error_analysis_v1.json`, held-out test set (10,660 rows / 23 matches), same
v1 fit as 8.2.

**By `phase_label`:** PR-AUC ranges from a low of 0.077 (`wide_defending_proxy`,
n=980, positive rate 2.0%) and 0.090 (`settled_mid_block_proxy`, n=1,896,
positive rate 2.6%) up to 0.506 (`box_defence`, n=1,694, positive rate 15.0%)
and 0.403 (`high_press_proxy`, n=1,775, positive rate 14.4%). The pattern is
consistent, not surprising: **PR-AUC tracks positive rate almost directly**
-- the phases with the rarest positives are also where the model is worst at
ranking them, which is the expected behaviour of a PR-based metric on
imbalanced slices rather than a phase-specific model weakness. Calibration is
good everywhere (gaps mostly <=0.012) except `transition_defence` (gap
+0.025, model over-predicts by about 60% relative to its 4.1% positive rate
in that slice) -- the one phase worth a closer look if this model is used
for per-phase probability thresholds.

**By `position`** (substituting for `defender_functional_role`, see above):
performance is broadly consistent across the 24 on-pitch positions (PR-AUC
mostly 0.28-0.61, ROC-AUC mostly 0.77-0.92), with the widest calibration gaps
on the smallest slices -- `Right Attacking Midfield` (n=62, gap -0.061,
under-predicting) and `Left Attacking Midfield` (n=66, gap -0.028) are both
under 70 rows and should be read as noisy, not as evidence of a real
attacking-midfield-specific bias. `Goalkeeper` (n=173) has the lowest positive
rate (1.7%) and correspondingly the most volatile PR-AUC (0.181) among
larger slices, consistent with the same rate-tracks-PR-AUC pattern seen in
`phase_label`. No position group shows a large, well-supported (n>200)
calibration gap in the way `transition_defence` does among phases.

### 8.5 Chart integrity

Automated pass, not a visual audit: all 16 expected PNGs
(4 variants &times; {`calibration_curve`, `precision_recall_curve`,
`roc_curve`, `prediction_distribution`}) exist, are non-empty, and are
pixel-dimension-consistent per chart type across all 4 variants (896&times;644
throughout, per the styling pass in the prior commit). No missing, near-zero,
unreadable, or inconsistently-sized files.

### 8.6 Bottom line

v1_unweighted's pick is well-supported: its CV edge over v2/v3 is real by a
paired t-test (and at the strongest non-parametric signal 5 folds can give),
it performs comparably across both tournaments actually present in the
dataset, and a genuine player-disjoint split reproduces its match-grouped CV
performance within noise -- reinforcing rather than undermining the Prompt 31
leakage-clearance conclusion. Its main known weakness is PR-AUC degrading on
low-positive-rate phase slices (`wide_defending_proxy`,
`settled_mid_block_proxy`), which is an expected consequence of class
imbalance within those slices rather than a defect specific to this model.
