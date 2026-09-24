# Dashboard data export (Prompt 83; regenerated Prompt 86)

> **Prompt 86 update.** This folder was originally exported in Prompt 83, then marked stale
> (`dashboard_data/STALE.md`, added at the end of Prompt 85) because Prompt 85 found and fixed a
> coordinate-frame bug in the loader (`src/dax/data/statsbomb_loader.py`) and re-ran the full
> feature/target/model pipeline on the correction. Every file in this folder has now been
> regenerated against that corrected pipeline, and the `STALE.md` marker has been removed. See
> [`reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md`](../reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md)
> for the full fix and re-validation, and section 8 below for exactly what changed in this folder.

Pre-computed JSON for the defensive-analytics dashboard page in the **separate `portfolio-website`
repo** (Next.js on Vercel). This folder is the whole handoff: it assumes no knowledge of the
project's history. **This repo produces data only.** There is no React/Next.js code here, and none
should be added.

Everything here is **re-scoring** of the six already-promoted reference models (same rung/family in
every case; none changed) on the corrected features. Two of the six legs
(`v1e_gradient_boosting_calibrated`, `d1_lognormal_glm`) still load their existing, never-retrained
`.joblib` artifacts, exactly as in Prompt 83, because Prompt 85's own validation reconfirmed their
numbers unchanged. The other four (`p1e_gradient_boosting_calibrated`, `c1d_random_forest`,
`x1c_random_forest`, `y1c_random_forest`) are refit at their already-locked hyperparameters (no new
tuning) on the corrected features, because Prompt 85's own Phase 4/5 validation scripts established
the corrected numbers this way and never persisted new `.joblib` artifacts to disk. See section 8.
No locked feature/target file was modified by this export, and no rung's own hyperparameters were
re-tuned.

---

## 1. Files

| File | What it is | Produced by |
|---|---|---|
| `match_explorer/3938643.json` | France 1-1 Poland, Euro 2024: every defensive moment, scored by all applicable models | `scripts/dashboard/export_match_explorer.py` |
| `match_explorer/3857294.json` | Netherlands 2-0 Qatar, WC 2022 | same |
| `match_explorer/3857298.json` | Portugal 3-2 Ghana, WC 2022 | same |
| `legs_summary.json` | The six models in plain English, one headline number each, grouped so binary and continuous legs are **never ranked together** | Hand-written from `reports/modeling/ALL_LEGS_SUMMARY.md`. Every number carries its `source` |
| `methodology_steps.json` | Pipeline walkthrough (7 stages) plus the "honest result" callouts | Hand-written from the ladder and closeout docs cited on each entry |
| `selection_and_frame_audit.json` | Style profile of all 23 held-out matches (the basis for picking the matches) and the coordinate-frame audit (section 5) | `scripts/dashboard/audit_frames_and_profile_matches.py` |

To regenerate (from the repo root, about 5-10 minutes; the passive legs load 1.6M rows):

```bash
.venv/Scripts/python.exe scripts/dashboard/audit_frames_and_profile_matches.py
```

```bash
.venv/Scripts/python.exe scripts/dashboard/export_match_explorer.py
```

**Size.** Each match file is 6.6-8.2 MB of minified JSON, or about 0.7-0.85 MB gzipped. Vercel
serves it gzipped. Most of the size is the passive defender rows (14.5k-18k per match). Load a
match file lazily when the user picks it, and don't bundle all three into the initial page.

---

## 2. How the matches were chosen (Task 1)

**Pool.** Only the 23 matches in the project's canonical **held-out test** split
(`outputs/models/splits/match_assignment.json`, value `"test"`). Every one is from WC 2022 or
Euro 2024. None of these matches was seen during training or model selection, so every prediction
in the explorer is **out-of-sample**.

**Criteria.** Each match needed a clear, visible difference in how the two teams defended, and the
three matches together needed some variety. The metrics come from each team's own defensive
actions, using the raw StatsBomb location in that team's own frame (own goal at x=0):

- mean action height (metres from own goal line)
- share of actions in the opponent's half, in the final third, and in the own third
- counterpress rate

**No model output and no target value was used to choose a match.** One caveat on that: the
first, exploratory profiling pass printed a per-match count of shot-within-10s windows alongside
the style numbers. That count was not a criterion, and the committed profiling script does not
compute it. Full numbers: `selection_and_frame_audit.json` →
`held_out_style_profile`.

### Chosen

| Match | Why | Style evidence (defending team's own actions, raw frame) |
|---|---|---|
| **France 1-1 Poland**, Euro 2024 group stage, 2024-06-25 (`3938643`) | Clearest **high press vs deep block** of all 23: the largest height gap (20.0 m) | France: mean 57.5 m from own goal, 55% of actions in the opponent half, 31% in own third. Poland: 37.5 m, 21%, 60% |
| **Netherlands 2-0 Qatar**, WC 2022 group stage, 2022-11-29 (`3857294`) | The WC example of the same contrast: 2nd-largest height gap (17.7 m) and the largest own-third gap (0.32) | Netherlands: 53.6 m, 22% of actions in the final third, 34% in own third. Qatar: 35.9 m, 4%, 66% |
| **Portugal 3-2 Ghana**, WC 2022 group stage, 2022-11-24 (`3857298`) | A different kind of game: open and transition-heavy, with the **highest counterpress rate of the 23** (0.205 match mean; the next highest is 0.17). Five goals. Portugal also defend high (27% of actions in the final third) | Portugal: 59.7 m, counterpress 0.20. Ghana: 45.7 m, counterpress 0.21 |

The three matches cover high press, deep block, and counterpress/transition defending. Every file
contains both active on-ball actions and passive off-ball positioning throughout.

### Rejected, and why

- **Slovenia 1-1 Serbia (`3930170`) was the original first pick, and it was wrong.** The first
  profiling pass used the stored `phase_label` and `action_x` columns, which made this look like
  the most extreme "Serbia press, Slovenia deep block" match in the pool. It turned out to be the
  coordinate-frame artefact described in section 5. Measured correctly, its height gap is 3.7 m,
  one of the smallest. Dropped, and every later selection used raw-frame metrics only.
- **Croatia 1-1 Brazil (`3869420`), WC 2022 quarter-final.** It was picked in the second pass
  (height gap 13.3 m, own-third gap 0.25) and fully exported. The export's own data-quality check
  then flagged it, and it was swapped out **on coverage grounds alone; no prediction values were
  looked at**:
  - 99.8% of its passive frames are `visibility_limited`, the highest of all 115 matches.
  - It has only 5.98 visible defenders per frame, the 3rd-lowest of 115 (the median is 8.2).
  - It includes extra time and 5 events from the penalty shootout (period 5).

  It would have been the weakest possible showcase for the three passive legs.
- **Ecuador 1-2 Senegal (`3857267`).** Thinnest 360 coverage in the test pool: 2,373 freeze-frames
  and 1,152 passive events, against roughly 3,100-3,300 and 1,600-2,200 for most test matches.
  Little style contrast either (5.5 m).
- **Netherlands 1-2 England (`3942819`), Euro 2024 semi-final.** Good contrast (16.5 m) and the
  best passive coverage, but the fewest active actions in the pool (299 rows; only 108 and 116 by
  the defending sides). That is too thin an active sample to showcase, and France-Poland already
  covers the same contrast for Euro 2024.
- **Germany 2-0 Hungary (`3930168`).** Strong contrast (15.1 m), but it is the same pattern from the
  same tournament as France-Poland, which has the larger gap. Redundant.

---

## 3. `match_explorer/{match_id}.json` schema

The top level follows the shape requested in Prompt 83, with a few additions:

```jsonc
{
  "schema_version": 1,
  "match_id": "3938643",
  "competition": "WC2022 | Euro2024",
  "competition_stage": "Group Stage",
  "teams": {"home": "France", "away": "Poland"},
  "score": {"home": 1, "away": 1},
  "date": "2024-06-25",
  "canonical_split": "test",                // always "test": every prediction is out-of-sample
  "why_selected": "...",
  "reference_models": {...},                // model id per leg
  "coordinate_frames": {...},               // human-readable description of the two location fields
  "data_quality": {...},                    // per-match counts, see section 6
  "events": [ ... ]                         // sorted by period, then timestamp
}
```

`events` holds two kinds of entry, told apart by `phase`.

### 3.1 `phase: "active"`: one on-ball defensive action

There is one entry per row of the active legs; the grain is one row per event.

```jsonc
{
  "event_id": "6fa63155-...",            // StatsBomb event UUID
  "timestamp": "00:18:42.859",           // StatsBomb timestamp, relative to the start of the period
  "minute": 18, "second": 42, "period": 1,
  "team": "France",                      // team of the player who made the action
  "player": "N'Golo Kanté",
  "phase": "active",
  "location": {"x": 108.8, "y": 52.9},   // acting team's OWN frame, see 3.3
  "location_match_frame": {"x": 108.8, "y": 52.9},
  "on_ball_event_type": "Ball Recovery", // the defensive action itself: Pressure, Duel, Interception, ...
  "counterpress": false,
  "won_possession": false,
  "phase_label": "transition_defence",   // rule-based proxy used as a MODEL INPUT, see section 5
  "phase_label_frame_consistent": false, // false = this row's label was computed in the wrong frame
  "predictions": {
    "active_binary":     {"probability": 0.6517},
    "active_continuous": {"expected_value": 0.04781, "expected_xg_given_shot": 0.07336},
    "active_xt":         {"expected_delta": 0.03663, "p_nonzero": 0.9381, "expected_delta_given_nonzero": 0.03905}
  },
  "observed": {"shot_within_10s": 0, "xg_within_10s": 0.0, "xt_delta": 0.2477}
}
```

### 3.2 `phase: "passive"`: one attacking on-ball event, with every visible off-ball defender

The passive legs score one row **per visible defender per attacking event** (the defender-slot
grain). The event-level fields would repeat identically on every defender row, so they are stored
once, and the defenders are nested under the event:

```jsonc
{
  "event_id": "a5c81a3d-...",
  "timestamp": "00:08:17.399", "minute": 8, "second": 17, "period": 1,
  "team": "France",                      // the DEFENDING team
  "player": null,                        // always null: see section 6
  "phase": "passive",
  "location": {"x": 46.9, "y": 49.7},    // the BALL, in the defending team's own frame
  "location_match_frame": {...},
  "on_ball_event_type": "Carry",         // the attacker's on-ball event
  "on_ball_team": "Poland",
  "visibility_limited": true,
  "phase_label": "settled_mid_block_proxy",
  "phase_label_frame_consistent": true,
  "defenders": [
    {
      "defender_slot_index": 0,          // index within this frame only; not a player id
      "location": {"x": 41.47, "y": 50.83},
      "location_match_frame": {...},
      "functional_role": "wide_cover",
      "predictions": {
        "passive_binary":     {"probability": 0.04528},
        "passive_continuous": {"expected_value": 0.004645, "expected_xg_given_shot": 0.1026},
        "passive_xt":         {"expected_delta": 0.001875, "p_nonzero": 0.5253, "expected_delta_given_nonzero": 0.00357}
      }
    }
  ],
  "observed": {"shot_within_10s": 0, "xg_within_10s": 0.0, "xt_delta": 0.007279}
}
```

`observed` is event-level: each defender row of an event carries the same target in the training
data. Every defender therefore gets their own prediction of the **same** outcome, made from their
own position. Nothing in this folder aggregates defenders into a single event-level prediction, and
no such aggregate was audited. If the page shows one, label it as a display aggregate (for example,
"mean across visible defenders").

### 3.3 Coordinates

- The pitch is StatsBomb's 120 × 80 units. Treat the units as yards, as StatsBomb specifies.
  (Section 2's "metres" are those same units, used loosely.)
- `location` is in the **acting team's own frame**: its own goal is at x=0 and the opponent's goal
  at x=120, so larger x means defending further up the pitch. For an active action, the acting team
  is the player's team. For a passive event and its defenders, it is the defending team. The same
  axis therefore means the same thing for both teams, which suits "how high does this team defend".
- `location_match_frame` puts both teams on one pitch for a whole-match map. The home team attacks
  toward x=120 and the away team toward x=0, in **both halves**. StatsBomb does not record the teams
  switching ends, so don't try to reproduce it.
- Both fields are rebuilt from **raw** StatsBomb locations, not from the model feature columns. See
  section 5 for why. The sanity check on the final files agrees: in all three matches, the
  high-defending team's mean own-frame x is higher than its opponent's in both the active and the
  passive data. For example, France averages 62.2 (active) and 59.7 (passive defenders); Poland
  averages 42.1 and 39.3.

### 3.4 Predictions: which fields appear where

| Sub-field | On | Model | Meaning |
|---|---|---|---|
| `active_binary.probability` | active events | `v1e_gradient_boosting_calibrated` | P(shot within 10 s) |
| `active_continuous.expected_value` | active events | `c1d_random_forest` × `v1e` P(shot) | Expected xG in the next 10 s, computed as `active_binary.probability × expected_xg_given_shot` |
| `active_continuous.expected_xg_given_shot` | active events | `c1d_random_forest` (log-normal corrected back-transform) | E[xG given a shot] |
| `active_xt.expected_delta` | active events | `x1c_random_forest` (two-stage) | Expected xT change, computed as `p_nonzero × expected_delta_given_nonzero` |
| `passive_binary.probability` | passive defenders | `p1e_gradient_boosting_calibrated` | P(shot within 10 s) |
| `passive_continuous.expected_value` | passive defenders | `d1_lognormal_glm` × `p1e` P(shot) | Expected xG, computed as `passive_binary.probability × expected_xg_given_shot` |
| `passive_xt.expected_delta` | passive defenders | `y1c_random_forest` (two-stage) | Expected xT change |

- **A missing key means "not applicable". It never means a predicted zero.** Active events never
  carry passive predictions and passive defenders never carry active predictions.
- The xT legs drop rows whose xT target could not be computed (active: 307 of 56,068 rows
  dataset-wide; passive: 2,581 of 1,593,181). Those rows have **no** `active_xt`/`passive_xt` key
  and no `observed.xt_delta`, because they sit outside the population the model was audited on. The
  per-match counts are in `data_quality`.
- The hurdle legs' `p_shot` is deliberately not repeated inside `*_continuous`. It is identical to
  the same side's `*_binary.probability`.
- Values are rounded to 4 significant figures for display. Coordinates are rounded to 2 decimals.
- `observed.xt_delta` is signed. Negative values are possible and meaningful (threat went down).

### 3.5 Provenance and verification of the predictions

`export_match_explorer.py` imports each leg's own scoring/fitting function from the script its
promotion audit used; no scoring code was rewritten for the export. **Updated in Prompt 86**: 2 of
the 6 legs still load a saved `.joblib` (never retrained); the other 4 are refit at their
already-locked hyperparameters on the corrected features, because Prompt 85's own Phase 4/5
validation established the corrected numbers this way and never persisted new `.joblib` artifacts
to the standing paths (see the top-of-file note above and this script's own module docstring):

| Leg | Scoring path used (Prompt 86) |
|---|---|
| Active-binary | `train_active_continuous_baseline.score_classifier` -- saved joblib, unchanged since Prompt 83 |
| Active-continuous | Prompt 86: `train_c1d_random_forest.fit_predict_c1d` (refit at locked hyperparameters) + `backtransform(..., "lognormal_corrected")` |
| Active-xT | Prompt 86: `train_x1c_random_forest.fit_predict` (refit at locked hyperparameters, on the Phase-3-corrected `target_xt_delta_v2`) |
| Passive-binary | Prompt 86: `train_p1e_gradient_boosting.fit_predict_gbm_calibrated` (refit, calibrated, at locked hyperparameters) |
| Passive-continuous | `train_passive_continuous_baseline.py` step [6/6] with the saved `d1_lognormal_glm.joblib` loaded, not refit. The script asserts the design-matrix width equals the artifact's `n_features_in_` |
| Passive-xT | Prompt 86: `train_y1c_random_forest.fit_predict` (refit at locked hyperparameters, on the Phase-3-corrected `target_xt_delta_passive`) |

Each model is first scored on its leg's **full** 23-match held-out set. In Prompt 83, the script
then hard-asserted bit-exact (1e-9) reproduction of each leg's pre-fix recorded held-out metric
before writing anything. **That hard gate no longer applies to the 4 refit legs** in Prompt 86 --
there is no persisted post-fix artifact to bit-exactly reproduce, and a fresh refit (even at
identical hyperparameters) carries its own internal non-determinism (LightGBM early stopping's
internal validation split, `CalibratedClassifierCV`'s internal folds). The script instead logs a
comparison against Phase 4/5's own corrected numbers (`phase4_held_out_refit.json`,
`phase5_xt_promotion_audit.json`) and does not hard-fail on a mismatch; the two untouched legs
(active-binary, passive-continuous) are logged against their pre-fix CSV/JSON records for the same
reason (corrected test-row features shift the score slightly even for an unchanged model). See the
script's own console output for the actual reproduced-vs-recorded numbers from the run that
produced the files in this folder.

Only after this comparison are the three curated matches sliced out. Event metadata (timestamp,
minute, second, acting team, raw location) comes from `data/raw/events/{match_id}.json`. Match
metadata comes from `data/raw/matches/*.json`. `observed.*` values are the locked targets in
`data/features/player_defensive_actions.parquet`, `data/features/passive_defense.parquet`, and
`outputs/prototypes/{active_binary_xt_delta_v2,passive_xt_delta}.parquet`.

**Passive-continuous reference, confirmed as current.** `MODELLING_CLOSEOUT.md` §2 and
`ALL_LEGS_SUMMARY.md` §2 both give `d1_lognormal_glm` as the standing reference. It was never
beaten; `d1d_random_forest` failed its gate at p=0.576. `d1_lognormal_glm` is what the export uses.

---

## 4. `legs_summary.json` and `methodology_steps.json`

### `legs_summary.json`

- `comparability.single_ranking_allowed: false`, `render_as: "separate_groups"`. **Render the three
  `groups` (binary / hurdle / xt) as separate cards or sections. Never sort all six legs into one
  list.** This carries forward `ALL_LEGS_SUMMARY.md` §5's finding: PR-AUC and R² are different kinds
  of number, and even the four R² legs sit on different target distributions and architectures.
- `legs[]`: `plain_name`, `question`, `model_family_plain`, one `headline` number
  (`metric` / `value` / `source`), `supporting_metrics`, `baseline_to_reference`, `audit` status, and
  a `one_liner`.
  - The binary legs also carry `random_guess_level`, the positive rate from the same CSV row. A
    PR-AUC of 0.43 means much more against a 7% base rate, and the page should say so.
- `active_vs_passive.by_family[]`: short strings per family, with
  `consistent_direction_across_families: false`. The honest headline is that the direction flips by
  family (`ALL_LEGS_SUMMARY.md` §4).
- `cross_leg_pattern`: random forest won 3 of 6 legs, and was tried on all 6 (`ALL_LEGS_SUMMARY.md`
  §3).

### `methodology_steps.json`

- `steps[]`: `order`, `stage`, a one-line `description`, optional `facts`, and a `source`.
- `honest_result_callouts[]`: the differentiators, each with `leg`, `step` (which stage it belongs
  to), `title`, `summary`, `numbers`, and `source`:
  - `x1d` and `y1d` not promoted
  - `y1b`'s bias finding, and the matching `x1b` finding
  - `d1`'s three independent null results
  - `c1e` not promoted
  - `v1e` rejected on slice calibration
  - the coordinate-frame issue found during this export

### Sources

Every number traces to the file named in its `source` field:

- `reports/modeling/ALL_LEGS_SUMMARY.md`
- `reports/modeling/MODELLING_CLOSEOUT.md`
- `reports/modeling/xt_target/XT_DELTA_CLOSEOUT.md`
- `reports/modeling/active_xt/ACTIVE_XT_MODEL_LADDER.md`
- `reports/modeling/passive_xt/PASSIVE_XT_MODEL_LADDER.md`
- `reports/modeling/passive_continuous/PASSIVE_CONTINUOUS_MODEL_LADDER.md`
- `reports/modeling/active_continuous/ACTIVE_CONTINUOUS_MODEL_LADDER.md`
- the CSV/JSON readouts under `outputs/models/`
- `selection_and_frame_audit.json`

---

## 5. Coordinate-frame issue: found in Prompt 83, fixed in Prompt 85

**Found while building this export in Prompt 83. Fixed in Prompt 85** (loader fix, commit
`8b7d0a8`, full re-pipeline in
[`reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md`](../reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md)).
**Full impact audit (Prompt 84)**:
[`reports/modeling/COORDINATE_FRAME_IMPACT_AUDIT.md`](../reports/modeling/COORDINATE_FRAME_IMPACT_AUDIT.md)
(`.html` version linked from `reports/modeling/INDEX.html`'s Project-wide group) is the complete,
traced-through-every-leg picture of which of the 6 targets were actually frame-dependent (not just
which features were) and the exact importance-weighted exposure share for each of the 6 promoted
reference models. This section is kept as the original Prompt 83 diagnosis, with the fix and its
effect noted inline below each affected number.

**The mismatch (pre-fix).** Raw StatsBomb event and 360 locations are always in the **acting team's
own frame**: the actor attacks toward x=120, whatever the period. Verified on a 40-match sample of
the raw event files here: 99.5% of 1,126 goalkeeper events sit at raw x<20, for both teams and both
halves. The project's loader used to assume the opposite, inferring a direction per
(period, possession team) from noisy ball-progression medians, which flipped one possession team
per period incorrectly.

**The fix (Prompt 85).** `src/dax/data/statsbomb_loader.py`'s `_infer_attack_sign_by_period_team`
was replaced with a row-local rule, `_attack_sign_for_row(team, possession_team)`: sign=+1 if the
row's own team is the possession team, else -1. Independently confirmed by a goalkeeper-event
sanity oracle: 92.8% of goalkeeper defensive-action events now sit at raw x<20 in their own frame
(up from the pre-fix loader's implicit assumption). The full feature/target/model pipeline was
then re-run once, all 6 legs re-validated (4 of 6 legs' headline held-out numbers moved; all 6 kept
their same promoted rung).

**The result, before vs after the fix:**

- On-ball events and defensive actions within the same possession used to land in **opposite**
  frames.
- Across `player_defensive_actions.parquet`, frame-inconsistent rows dropped from **26,039 of
  56,068 (46.4%)** pre-fix to **2,709 of 56,068 (4.83%)** post-fix (source:
  `selection_and_frame_audit.json` → `frame_audit`, this export's own regenerated numbers).
- `passive_defense.parquet` was affected the same way, event by event, and improved the same way.

**The rule-based `phase_label` inherited the problem** (still does, to a much smaller degree),
because it reads `ball_x` and treats x ≥ 95 as box defence and x ≤ 30 as high press:

| Clearances (dataset-wide) | Labelled `box_defence` | Labelled `high_press_proxy` |
|---|---|---|
| Frame-consistent rows (pre-fix and post-fix, similar) | ~75-79% | 0.7-0.9% |
| Frame-inconsistent rows (pre-fix, 46.4% of rows) | 1.0% | 72.4% |

Post-fix, only 4.83% of rows are frame-inconsistent at all (vs 46.4% pre-fix), so the
frame-inconsistent Clearance-row mislabelling this table describes now affects a much smaller
population; see `reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md` Phase 2 for the full
Clearance/phase_label sanity re-check (dataset-wide `high_press_proxy` share among Clearances
dropped from 30.3% to 0.9%, `box_defence` rose from 44.6% to 72.8%).

Source: `selection_and_frame_audit.json` → `frame_audit`, reproducible with
`scripts/dashboard/audit_frames_and_profile_matches.py` (re-run against the corrected pipeline for
this export).

**What it means:**

- **Model metrics (Prompt 83, pre-fix).** Every model was trained, cross-validated, and tested on
  the same features, so the held-out numbers were real measurements of these models on those
  features. But the spatial features, and every `phase_label`-sliced analysis in the reports, were
  noisier or partly mislabelled compared with what was intended.
- **Model metrics (Prompt 85/86, post-fix).** Confirmed, not hypothetical: 4 of the 6 promoted
  reference models' headline held-out numbers moved once trained/evaluated on corrected features
  (2 xT legs' targets were themselves recomputed; 2 non-xT legs' promoted models improved on
  corrected features), while 2 legs' numbers were reconfirmed unchanged (within noise). All 6 legs
  kept their same promoted rung/architecture. See
  `reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md` for the full per-leg before/after.
- **This dashboard.** `location` and `location_match_frame` are rebuilt from raw locations, so the
  pitch plots have always been correct, pre- and post-fix. `phase_label` is still exported, because
  it is a real model input, but every event carries `phase_label_frame_consistent`. **Don't show
  `phase_label` as a tactical fact when that flag is `false`**; hide it or grey it out. Each file's
  `data_quality` counts the affected rows -- post-fix, this count is much smaller (dataset-wide,
  4.83% of active rows are frame-inconsistent, down from 46.4%).
- **Done since Prompt 85.** The loader was fixed and the full feature → model pipeline was re-run
  (Prompt 85); this export (Prompt 86) re-scores against that corrected pipeline. No locked file
  was modified by this export itself.

---

## 6. Other gaps the frontend should know about

- **No player identity for passive defenders.** StatsBomb 360 freeze-frames are anonymous. That is
  why `player` is always `null` and `defender_slot_index` is a per-frame index, not a player id.
  Don't label passive dots with names. Active actions do carry real player names.
- **Visible defenders only.** The passive side only sees players inside the 360 camera's view. The
  count per event varies:

  | Match | Defenders per event (min / mean / max) |
  |---|---|
  | France-Poland | 2 / 8.57 / 11 |
  | Netherlands-Qatar | 1 / 8.25 / 11 |
  | Portugal-Ghana | 1 / 8.17 / 11 |

- **`visibility_limited` share of passive events.** France-Poland 0.842, Netherlands-Qatar 0.978,
  Portugal-Ghana 0.802, against a dataset-wide row share of 0.833. Netherlands-Qatar's 360 view is
  narrower than typical, although its defender counts per frame are normal. That was the deciding
  difference from the rejected Croatia-Brazil.
- **Frame-inconsistent rows per match** (section 5). **Prompt 86 update: these counts dropped
  sharply after the coordinate-frame fix** (compare to the pre-fix counts, kept below for
  reference):

  | Match | Active rows (post-fix / pre-fix, both /total) | Passive events (post-fix / pre-fix, both /total) |
  |---|---|---|
  | France-Poland | 21/500 / 258/500 | 60/1,699 / 907/1,699 |
  | Netherlands-Qatar | 23/463 / 226/463 | 75/2,194 / 1,415/2,194 |
  | Portugal-Ghana | 25/443 / 223/443 | 50/1,775 / 713/1,775 |

- **Timestamps restart each period.** `timestamp` is relative to the start of the period
  (StatsBomb convention). Use `period` + `minute` + `second` for a single match clock.
- **Passive events with no raw location.** These would be dropped and counted in
  `rows_with_unknown_frame`. There are 0 in all three files.

---

## 7. Notes for the record

- **The plan doc is missing.** The Prompt 83 brief references
  `claude/eda-prompts/82-portfolio-dashboard-plan.md` (§3.1 / §3.3). That file does not exist in
  this repo or anywhere under the user's home directory. `legs_summary.json` and
  `methodology_steps.json` follow the brief's own description of those sections: plain-English leg
  names, one headline number per leg, active/passive as short strings, and stage + one-liner +
  honest-result callouts. If the plan doc turns up and differs, reconcile against it.
- **The p1e "discrepancy" in `ALL_LEGS_SUMMARY.md` §2 is not a discrepancy (pre-fix numbers; see
  Prompt 86 below for the current headline).** That section reported the CSV as showing PR-AUC
  0.210773 for `p1e`, against the doc's pre-fix 0.2162. Re-checked in Prompt 83: 0.210773 was the
  **uncalibrated** `p1e_gradient_boosting` row; the calibrated reference row read 0.2161872,
  matching the doc. `ALL_LEGS_SUMMARY.md` was not edited in Prompt 83; it was edited in Prompt 86
  (see below) to cite the corrected 0.2267.

---

## 8. Prompt 86: what changed in this regeneration

Regenerated against Prompt 85's corrected coordinate-frame pipeline (full context:
[`reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md`](../reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md)).
Re-ran this folder's own documented export procedure (`scripts/dashboard/audit_frames_and_profile_matches.py`
then `scripts/dashboard/export_match_explorer.py`), scoring the **same 3 curated matches** -- no
match was re-selected, added, or removed.

- **`selection_and_frame_audit.json`**: fully regenerated. `frame_audit.share_frame_inconsistent`
  drops from 0.4644 (pre-fix) to 0.0483 (post-fix); `held_out_style_profile` is naturally recomputed
  from corrected `ball_x`/`ball_y`, though the three curated matches' own selection rationale
  (section 2 above) was not re-evaluated -- the task's own instruction was to re-score the already-
  curated matches, not re-select them.
- **`legs_summary.json`**: 4 of 6 legs' headline numbers updated (passive-binary, active-continuous
  xT-adjacent context, active-xT, passive-xT); active-binary and passive-continuous numbers
  unchanged (Phase 4 reconfirmed them, within noise). The `active_vs_passive.by_family` xt row's
  **conclusion reverses**: active's own gain over its Rung 0 is now larger than passive's on the
  corrected target (previously the opposite was reported, scored against a target that was itself
  wrong for 50-69% of its own rows).
- **`match_explorer/{match_id}.json`** (all 3 files): every leg's predictions were re-scored.
  2 legs (`v1e_gradient_boosting_calibrated`, `d1_lognormal_glm`) reuse their existing, untouched
  `.joblib` artifacts scored on corrected features. 4 legs (`p1e_gradient_boosting_calibrated`,
  `c1d_random_forest`, `x1c_random_forest`, `y1c_random_forest`) are refit at their already-locked
  hyperparameters on corrected features (and, for the 2 xT legs, the corrected target) -- see
  `scripts/dashboard/export_match_explorer.py`'s own module docstring and section 3.5 above for why
  no persisted post-fix artifact existed to load instead. `data_quality` counts (frame-inconsistent
  rows per match) dropped sharply; see section 6 above for the regenerated per-match table.
- **`methodology_steps.json`**: the `coordinate_frame_issue` callout was rewritten to describe the
  fix (was: describes the still-open bug with pre-fix numbers).
- **`dashboard_data/STALE.md`**: deleted once this regeneration was verified complete and correct.
