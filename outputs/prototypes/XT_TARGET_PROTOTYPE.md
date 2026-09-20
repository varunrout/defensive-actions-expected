# xT-Delta Target Prototype (Prompt 64) — Active-Binary Leg Only

> **Superseded (Prompt 66):** this document's `xt_after` definition (`xT(action_x, action_y)` — the
> defensive action's own recorded location) has been replaced by a possession-outcome definition
> (value of the next same-possession event, or 0 if the action ends the possession) that fixes the
> Clearance artifact section 3.1/3.4 below describe rather than merely documenting it. See
> [`XT_TARGET_PROTOTYPE_V2.md`](XT_TARGET_PROTOTYPE_V2.md) for the corrected methodology and full
> re-validated results. This document is kept as-is, not deleted or rewritten — it remains the
> accurate record of what Prompt 64 actually did and found at the time.

Prototype only. Not a committed project report, no HTML twin. Scoped to the active-binary leg
(`data/features/player_defensive_actions.parquet`) per the agreed prototype-first plan — the passive
leg and both continuous legs are untouched. `player_defensive_actions.parquet` and
`src/dax/targets/short_horizon.py` are both read-only in this prototype; nothing locked was modified.

## 1. Tooling

**`socceraction` does not install/import cleanly in this repo's real `.venv`** (Python 3.13.3, numpy
2.4.6, pandas 2.3.3) — checked directly, not assumed, and this is reported plainly rather than forced
through or silently worked around with a different interpreter (the exact mistake Prompt 58 already
made and fixed once this session).

- `pip install socceraction` (with dependency resolution) attempts to build **numpy from source**
  (no compiler available on this machine) because `socceraction==1.4.2` (latest PyPI release, confirmed
  via `pip index versions`) pins `pandas<2.0.0` and `pandera<0.14.0,>=0.13.4`.
- Installed with `--no-deps` instead (`socceraction==1.4.2` now present in `.venv`, confirmed
  importable at the package level). Its actual dependency, `pandera` in the `<0.14.0` range it
  requires, **fails to import under numpy>=2.0**: `AttributeError: np.string_ was removed in the
  NumPy 2.0 release`. A newer, numpy-2.x-compatible `pandera` (0.33.1) imports fine on its own, but
  socceraction's own SPADL schema classes (`socceraction/spadl/schema.py`) use `pa.SchemaModel`, an
  API `pandera` 0.33.1 has since removed (`AttributeError: module 'pandera' has no attribute
  'SchemaModel'`).
- **This is a genuine three-way version deadlock** (socceraction 1.4.2 / pandera<0.14 / numpy>=2.0),
  not a typo or a missing flag. Downgrading numpy/pandas project-wide to satisfy `socceraction`'s
  2021-era pins was **not attempted** — that risks breaking the other three legs' already-promoted
  models and the full 248-test suite, an unacceptable blast radius for a prototype. Confirmed the
  main `.venv`'s numpy/pandas/sklearn/lightgbm versions are unchanged from before this prompt
  (`numpy-2.4.6.dist-info` timestamp predates this session; no regression introduced).

**Actual API investigated directly against the installed 1.4.2 source**
(`.venv/Lib/site-packages/socceraction/xthreat.py`), not from memory or docs: the module defines
`ExpectedThreat` (a Markov-chain grid fitter, `M=12` width cells x `N=16` length cells by default)
and a `load_model(path)` function whose own docstring states plainly: *"Karun Singh provides such a
grid at the following url: https://karun.in/blog/data/open_xt_12x8_v1.json"* — `load_model` itself
is nothing more than `pd.read_json(path)` wrapped in the class. **`socceraction` does not ship a
pretrained grid as package data** — it documents where to fetch the original, publicly published
Karun Singh grid (the xT framework's original author, https://karun.in/blog/expected-threat.html)
and loads it from there. Per Varun's explicit decision this session (use the public/pretrained grid,
not a from-scratch fit on this repo's own two-tournament data), that URL's grid — an **8x12** array
(8 width cells, 12 length cells; different from `ExpectedThreat`'s own 12x16 fitting default, which
only applies when fitting a *new* grid) — was fetched once and vendored locally at
`src/dax/targets/open_xt_12x8_v1.json` so this module never needs network access at runtime.

Because the full `socceraction` package cannot actually be imported here, `expected_threat.py` does
**not** import it — it is a direct, minimal reimplementation of `socceraction.xthreat`'s own
`_get_cell_indexes`/`ExpectedThreat.rate` indexing convention (read from the installed source before
writing this module), applied to the same public grid `socceraction.xthreat.load_model` itself
documents. `socceraction` remains installed in `.venv` for provenance/documentation purposes only.

**Coordinate transform, confirmed not assumed.** This project's own pitch convention (confirmed
directly against `player_defensive_actions.parquet`'s real `action_x`/`action_y` range: 0.2-119.9 /
0.0-80.0) is a 120x80-yard pitch, attacking goal at (120, 40). SPADL (socceraction's own convention,
and what the Karun Singh grid is built against) is a 105x68-metre pitch. Both put the attacking goal
at the "high x" end with the same y-orientation, so the transform is a pure per-axis rescale
(`x_spadl = x_project / 120 * 105`, `y_spadl = y_project / 80 * 68`) — no flip or rotation needed.
Implemented in `_project_to_spadl()`.

## 2. Deliverables

- `src/dax/targets/expected_threat.py` — the standalone lookup module (grid loaded once via
  `functools.lru_cache`, not re-fit per call).
- `src/dax/targets/open_xt_12x8_v1.json` — the vendored public grid (Karun Singh, fetched once this
  session).
- `scripts/analysis/build_xt_delta_prototype.py` — computes `xt_before`/`xt_after`/`target_xt_delta`
  for all 56,068 active-binary rows.
- `outputs/prototypes/active_binary_xt_delta.parquet` — 56,068 rows, 4 columns
  (`event_id`, `xt_before`, `xt_after`, `target_xt_delta`), row-aligned to the locked parquet by
  `event_id`, **not merged into it**.

**`xt_before`/`xt_after` definitions, confirmed against real data before computing anything:**
`player_defensive_actions.parquet`'s `action_x`/`action_y` were confirmed **identical to
floating-point exactness** to `data/processed/events_with_targets.parquet`'s own `ball_x`/`ball_y` at
the same `event_id` (100% row match, correlation 1.0, mean absolute difference 0.0) — so `xt_after`
uses `action_x`/`action_y` directly, per the task's own definition. `xt_before` uses the **previous
event's** `ball_x`/`ball_y` (same `match_id`, `index = this row's index - 1`) —
`events_with_targets.parquet`'s `index` column is confirmed contiguous per match (no gaps, checked
directly), so this is a real previous-event lookup, not an approximation. 32 of 56,068 rows (0.06%)
have `xt_before = NaN` because the immediately preceding event is a non-ball event with no location
at all (`Tactical Shift` x15, `Player On` x10, `Substitution` x4, `Injury Stoppage` x2, `Player Off`
x1 — checked directly, not a bug) — left as NaN, not silently zero-filled.

## 3. Validation — the real deliverable

### 3.1 Does it fix the degenerate-zero problem? **Yes, directly confirmed.**

| Slice | Old target (`target_future_shot_10s`) | New target (`target_xt_delta`) |
|---|---|---|
| `action_ended_possession == True` (n=4,662 / 4,656 defined) | mean **0.000000**, 100% identical | mean **-0.00121**, std 0.0680, 23.5% exactly zero, 40.7% negative, 35.8% positive |
| `action_won_possession == True` (n=3,605 / 3,600 defined) | mean **0.000000**, 100% identical | mean **-0.00352**, std 0.0703, 21.0% exactly zero, 42.8% negative, 36.2% positive |

The tautological 100%-identical-zero collapse is gone: both slices now show real, row-level variance
(roughly 60-65% of rows in each slice are non-zero) rather than a single repeated value. **This is
the core problem the prototype was built to fix, and it is fixed** -- the new target can
distinguish "possession resolved in the defense's favour" from other outcomes at the row level,
which the old target structurally could not.

**But the average sign within these "defense succeeded" slices leans slightly negative, not
positive** -- worth flagging honestly rather than only reporting that the zero-collapse is fixed.
Breaking `action_ended_possession` down by `event_type` explains why, and surfaces a genuine
definitional subtlety in this prototype, not a bug in the grid or the lookup:

| `event_type` | n | mean `target_xt_delta` |
|---|---|---|
| Ball Recovery | 2,471 | -0.0004 |
| Interception | 522 | -0.0019 |
| Block | 457 | +0.0023 |
| Duel | 420 | +0.0023 |
| Pressure | 418 | +0.0076 |
| **Clearance** | **289** | **-0.0315** |
| Foul Committed | 42 | +0.0027 |
| 50/50 | 37 | +0.0009 |

**`Clearance` is the one event type driving the negative mean, by a wide margin.** Spot-checked
directly (section 3.4): for a `Clearance`, `action_x`/`action_y` is the location the defender
performed the clearance **from**, not where the ball ends up after the clearing kick. A last-ditch
clearance is very often taken deep in the defending team's own box (`action_x` frequently 108-118 in
this leg's own data, i.e. very close to the "attacking goal" end of the per-action-normalized
coordinate frame) -- a **high-xT location by construction**, regardless of how well the defensive
action itself resolves the danger, because xT only knows "how threatening is continued possession
at this exact spot," not "the ball is about to be booted 40 metres away." This is a real, reportable
limitation of using `action_x`/`action_y` (the action's own location, per this prototype's task
definition) as `xt_after` for clearances specifically -- not something this prototype fixes, since
that was the task's explicit definition of `xt_after`, but flagged plainly as a design question for
any full rollout: a clearance's *resulting* ball location (where it lands) would likely be a more
football-accurate `xt_after` than the clearance's own action location.

### 3.2 Distribution shape

On the 56,036 rows with a defined `target_xt_delta` (32 NaN, section 2):

| Statistic | Value |
|---|---|
| Mean | -0.00132 |
| Std | 0.06499 |
| Skew | +0.093 (nearly symmetric) |
| Excess kurtosis | 8.85 (heavy-tailed -- most mass near zero, a few large outlier swings) |
| Share exactly zero | 11.3% |
| Share negative | 47.0% |
| Share positive | 41.7% |

**This is a fundamentally different shape from the old continuous target
(`target_future_xg_10s`)**, which was strictly non-negative, zero-inflated, and heavily right-skewed
(the whole reason the active-continuous leg's hurdle architecture exists). `target_xt_delta` is
**roughly symmetric and can be negative** -- nearly half the rows are negative (the defensive action
coincided with the attack becoming *more* dangerous, not less). **The current hurdle architecture
(P(shot) x E[xg|shot], built specifically for a zero-inflated, positive-skewed target) likely does
not apply as-is to this target.** A single continuous regression (no hurdle split, possibly still
log-scale-adjacent given the heavy kurtosis, or a direct location-scale model) may be the right
structure instead -- **flagged here as an open design question, not decided in this prototype.**

### 3.3 Correlation with the old target

On rows where `target_xt_delta` is defined: Pearson **r = -0.016**, Spearman **r = -0.015** (both
essentially zero, whether restricted to the non-degenerate subset (n=51,380, excluding
`action_ended_possession`/`action_won_possession` rows) or computed over all 56,036 defined rows).
**The two targets are practically uncorrelated.** This is not itself a red flag -- they measure
genuinely different things by construction: `target_future_xg_10s` is forward-looking (attacking
output in the *next* 10 seconds), `target_xt_delta` is an instantaneous value-swing *at* the action
itself. Near-zero correlation confirms the new target is not simply a relabelled copy of the old
one, but it also means **no assumption should be carried over** that whatever this leg's models
learned about `target_future_xg_10s` would transfer to `target_xt_delta`** -- consistent with the
task's own framing that a full leg redo requires retraining, not fine-tuning.

### 3.4 Spot-check, 10 rows, not cherry-picked for a favourable story

Five from the extreme ends of the distribution (the actual largest-magnitude rows, both directions)
plus five representative event-type examples from section 3.1's breakdown:

| `event_id` (short) | `event_type` | `action_x`,`y` | `xt_before` | `xt_after` | `target_xt_delta` | Read |
|---|---|---|---|---|---|---|
| `52bc252a…` | Ball Recovery | 9.8, 2.6 | 0.2575 | 0.0064 | **+0.2511** | Largest positive swing in the dataset -- ball recovered literally on the defending team's own goal line, immediately after a maximal-threat previous event (almost certainly a save/block rebound). Football-sensible: the most extreme "danger just averted" case possible. |
| `115479c8…` | Duel | 7.1, 74.7 | 0.2575 | 0.0064 | **+0.2511** | Same pattern -- duel won right in front of goal after a maximal-threat prior state. Sensible. |
| `53e9e817…` | Clearance | 110.8, 36.9 | 0.0064 | 0.2575 | **-0.2511** | Largest negative swing -- previous event was a `Pass` from `(0, 0)` (own corner/placeholder-adjacent location), clearance taken deep in the box. Per section 3.1, this is the `action_x`-as-`xt_after` artifact, not a genuinely worse outcome -- the clearance itself is a good defensive action, mismeasured by this prototype's location choice. |
| `6985cc1e…` | Clearance | 114.3, 38.8 | 0.0064 | 0.2575 | **-0.2511** | Same pattern; previous event `Pass` from `(0, 79.9)` -- exact pitch corner, worth a data-quality follow-up question (genuine deep pass, or a coordinate placeholder) before any full rollout, not resolved here. |
| `73a56426…` | Interception | 74.4, 4.7 | 0.0121 | 0.0175 | -0.0053 | Small negative, not the "clean positive" the task's own working hypothesis expected -- interceptions rarely move the ball far from where it already was heading, so `xt_before`/`xt_after` often land in the same or an adjacent low-value cell. Sensible once understood, but not the intuition-confirming example one might expect going in. |
| `09f08c32…` | Interception | 95.5, 57.2 | 0.0286 | 0.0286 | 0.0000 | Same-cell interception -- delta exactly zero. Grid resolution (8x12 cells over 105x68m, ~13x8.5m/cell) is coarse enough that many interceptions don't cross a cell boundary at all. |
| `979acbe5…` | Clearance | 5.8, 35.1 | 0.0644 | 0.0094 | **+0.0550** | A clearance where `action_x` (5.8, deep in the *defending* end of this normalized frame) happens to coincide with a genuinely lower-threat spot than the prior state -- positive, sensible, and the counter-example showing section 3.1's Clearance bias isn't universal, just directionally dominant. |
| `de6330cd…` | Foul Committed | 68.5, 56.0 | 0.0161 | 0.0169 | -0.0007 | Small negative -- a foul conceded in a moderately advanced, non-central area. Plausible: gives the attack a restart chance from a slightly more dangerous spot than the immediately preceding phase of play, though the magnitude is small since neither location is especially dangerous. |
| `9a064510…` | Foul Committed | 23.2, 66.0 | 0.0084 | 0.0094 | -0.0010 | Same small-negative pattern, foul conceded near the defending team's own corner -- low absolute threat throughout, small delta, sensible. |
| `5fa05f03…` | Clearance | 47.9, 46.4 | 0.0169 | 0.0126 | +0.0043 | A clearance from a central, moderate-threat area (not deep in the box) shows a small positive delta -- consistent with section 3.1's explanation: the `action_x`-as-`xt_after` bias is specifically a deep-box-clearance artifact, not a property of clearances generally. |

## 4. Honest conclusion — go/no-go evidence, not a decision

This prototype's job was to produce evidence, not make the call (Varun's, after reading this):

- **Does it fix the degenerate-zero problem?** Yes, unambiguously -- the two motivating slices go
  from 100.0% identical zeros to real, row-level-varying distributions (60-65% non-zero rows in
  each).
- **Is the distribution tractable for modelling as-is?** Partially. It's well-behaved in the sense of
  having a large sample, no missing-data surprises beyond the 32 explained NaN rows, and a
  football-sensible spread of magnitudes -- but it is **structurally incompatible with the current
  hurdle architecture** (roughly symmetric, ~47% negative, not zero-inflated-positive-skewed). A
  full rollout would need a different model structure for the continuous legs specifically, decided
  separately from this prototype.
- **Is it football-sensible on inspection?** Mostly yes, with one clear, specific, non-trivial
  caveat found and reported rather than glossed over: `action_x`/`action_y` (this prototype's
  required definition of `xt_after`) systematically understates the value of deep-box clearances,
  because it evaluates threat at the clearance's own action location rather than the ball's
  resulting location after the clear. This drags the `action_ended_possession` slice's average sign
  slightly negative even though the underlying defensive actions are, in football terms, good
  outcomes.
- **Does the new target correlate with the old one?** No, essentially zero correlation -- expected
  given they measure different things (instantaneous value-swing vs. forward-looking future
  outcome), but a real finding: nothing about this leg's existing model results can be assumed to
  transfer, a full retrain really would be required, matching the task's own framing of the cost.

**Net read**: the core motivating problem is genuinely fixed, but this prototype also surfaces a
concrete, fixable definitional issue (the clearance/`action_x` mismatch) and a real structural
question (hurdle vs. single-regression) that a full four-leg rollout would need to resolve first --
neither is a reason to reject the xT approach outright, but neither is trivial enough to wave past.
This is the evidence; the go/no-go call on the full rollout is Varun's.
