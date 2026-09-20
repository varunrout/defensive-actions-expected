# Passive xT-Delta Target Prototype (Prompt 68) — Possession-Outcome Definition From The Start

Prototype only. Not a committed project report, no HTML twin. Passive leg only — no active-leg file
or output is touched. `passive_defense.parquet`, `src/dax/targets/short_horizon.py`, and
`src/dax/features/event_context.py` are all read-only throughout; nothing locked was modified. No
model artifact retrained or touched.

**Why this leg skips the v1/v2 detour active-binary needed (Prompts 64/66/67).** Active's mistake was
using the defensive action's *own recorded location* (`action_x`/`action_y`) as `xt_after` — the
spot the defender happened to be standing, not where the ball actually went. That mistake never had
anywhere to happen here: `passive_defense.parquet`'s `ball_x`/`ball_y` is already the ball's location
*at the on-ball event the snapshot is anchored to* (confirmed identical to
`events_with_targets.parquet`'s own `ball_x`/`ball_y` at the same `event_id`, to floating-point
exactness, section 0), which is the correct "before" state by the exact same reasoning Prompt 66
established for active's `xt_before`. This prototype implements the corrected, possession-outcome
`xt_after` definition directly, on the first pass.

## 0. Confirmed schema, before writing anything

**Same underlying event stream as active.** `passive_defense.parquet`'s 198,354 unique `event_id`
values match 100% onto `events_with_targets.parquet`'s own `id` column (checked directly, 0
unmatched) — same `match_id, period, index` canonical ordering Prompt 66 used for active, no separate
infrastructure needed.

**`action_ended_possession` is a general event property, not defensive-action-specific.** Checked
directly: it is non-null for all 422,561 rows of `events_with_targets.parquet` (every event, not just
the active-binary defensive-action subset that dataset was originally built for) — `add_event_context()`
computes it once over the full event stream. Reused unchanged, no fresh computation needed for the
passive leg.

**Shared-target precedent, confirmed empirically, not assumed.** `target_future_shot_10s` and
`target_future_xg_10s` were checked directly for every one of the 198,354 unique `event_id` values in
`passive_defense.parquet`: **0 events show more than one distinct value** for either target across
the defender rows that share that `event_id` (mean rows/event ~8.03 here, though this varies by
event — see below). This is the precedent `target_xt_delta_passive` follows: computed once per event,
joined onto every defender row.

**Next-event location and Shot handling transfer unchanged.** Same source data, same lookup logic as
Prompt 66 (`match_id` + `period`, `index + 1`, period-boundary-respecting). Checked directly against
the 198,354 unique passive events: 1,619 (0.82%) have `action_ended_possession == True`; of the
196,735 that continue, **863 (0.44%) have a Shot as the next event** (`shot_statsbomb_xg` populated
for 100% of these, 0 NaN) — valued via that xG, not a grid lookup, exactly Prompt 66's convention.
**0 rows** are the last event of their period with `action_ended_possession` still `False` (the
period/match-end edge case) — same clean result as active, no fallback branch needed in practice.
345 of the 196,735 continuing events (0.18%) have a next event with no populated `ball_x`/`ball_y` —
left as `NaN`, not zero-filled.

**Everything here transferred cleanly — no fallback or approximation was needed anywhere in item 0.**

## 1. Implementation

`scripts/analysis/build_passive_xt_delta_prototype.py`. Per unique `event_id`:

- `xt_before = xT(ball_x, ball_y)` at the snapshot itself — no previous-event lookup (unlike active).
- `xt_after = 0.0` if `action_ended_possession == True`; otherwise the next same-possession event's
  `xT(ball_x, ball_y)`, or that event's own `shot_statsbomb_xg` if it is a Shot.
- `target_xt_delta_passive = xt_before - xt_after`.

Computed **once per unique `event_id` (198,354 rows)**, not per defender row — there is nothing
defender-specific in this target's definition. Written to
`outputs/prototypes/passive_xt_delta.parquet` (`event_id`, `xt_before`, `xt_after`,
`target_xt_delta_passive`) at this same unique-event grain — meant to be **left-joined onto
`passive_defense.parquet` by `event_id` downstream**, the same way `target_future_shot_10s`/
`target_future_xg_10s` are already one value per event repeated across every defender row that
shares it, not pre-expanded to the full 1,593,181-row grain in this prototype file.

## 2. Validation

### 2.1 Possession-ending subset — collapses to `xt_before`'s real distribution, as expected

| Slice | n (all defined) | `target_xt_delta_passive` mean | `xt_before` mean (same slice) | Identical? |
|---|---|---|---|---|
| `action_ended_possession == True` | 1,619 | **+0.02452** | +0.02452 | **Yes, exactly** (`np.allclose`) |

Same logically-required property Prompt 66 confirmed for active: `xt_after` forced to `0.0` means the
delta collapses to `xt_before`'s own real, varying distribution — not a synthetic zero. Consistent
with active's own possession-ending numbers (+0.0305/+0.0307 there) in sign and rough order of
magnitude, smaller here (this leg's on-ball events are far more often mid-buildup passes/carries than
active's own defensive actions, so the average denied threat at the point a possession happens to end
is lower).

### 2.2 Distribution (unique-event level, 198,009 defined values)

| Statistic | Value | Active v2, for reference |
|---|---|---|
| Mean | **-0.00181** | +0.00338 |
| Std | 0.03828 | 0.05838 |
| Skew | **-1.779** | -0.306 |
| Excess kurtosis | **31.02** | 18.33 |
| Share exactly zero | 26.3% | 20.2% |
| Share negative | 36.9% | 35.4% |
| Share positive | 36.9% | 44.4% |

**Mean is small and slightly negative here, not positive like active's.** This is a real, sensible
difference in what's being measured, not a discrepancy to be alarmed about: active-binary rows are
specifically *defensive actions* (interventions that, on average, should deny threat); passive rows
are snapshots taken at an **arbitrary on-ball event during a live attacking phase** — there is no
inherent reason attacking possession should lose value on average from one touch to the next; if
anything, a slightly negative mean (threat creeping up slightly as attacks progress) is the more
plausible prior. **Heavier left tail and far higher kurtosis than active** (skew -1.78 vs -0.31,
kurtosis 31 vs 18) — same mechanism as active (shot-xG injection for the 863 next-is-Shot events,
up to close to 1.0), but relatively more concentrated here since possession-ending events are a much
smaller share of this leg's own event population (0.82% vs active's much larger flagged share) while
the shot-xG-tail rows are a comparable small fraction either way — worth flagging plainly for any
future modelling decision on this leg, same as Prompt 66 flagged for active: **this target is not
close to the old zero-inflated positive-skewed shape either, and is if anything heavier-tailed than
active's own already-heavy-tailed version.**

### 2.3 Correlation with the old targets

| Pair | Pearson r |
|---|---|
| `target_xt_delta_passive` vs `target_future_shot_10s` | **-0.0629** |
| `target_xt_delta_passive` vs `target_future_xg_10s` | **-0.0715** |

**Same direction as active v2's improved relationship (-0.108 vs the old target), correctly signed,
but numerically weaker here** — stated plainly, not talked up. More threat denied is still associated
with less future attacking output, which is the sensible direction, but the relationship is looser on
this leg. A plausible reason, not chased further in this prototype: a passive snapshot is one
defender's off-ball positioning at an arbitrary point in a live attack, one step further removed from
the moment-to-moment outcome than active's own defensive *actions* are, so a weaker link between
"value denied at this instant" and "what happens in the next 10 seconds" is not surprising.

### 2.4 Spot-check, 10 rows, unique events

| `event_id` (short) | `on_ball_event_type` | Ended? | `xt_before` | `xt_after` | delta | Read |
|---|---|---|---|---|---|---|
| `76adcbc4…` | Pass | Yes | 0.0138 | 0.0000 | +0.0138 | Possession ends right after this pass -- full credit for whatever threat existed, correctly denied. |
| `0562fc11…` | Shot | Yes | 0.0094 | 0.0000 | +0.0094 | A shot that itself ends the possession (no rebound continuation) -- small `xt_before` since the snapshot's own location wasn't especially dangerous by grid terms. |
| `29f0887e…` | Pass | Yes | 0.0108 | 0.0000 | +0.0108 | Same pattern, ordinary case. |
| `433267d8…` | Carry | Yes | 0.0148 | 0.0000 | +0.0148 | A dribble/carry that ends possession immediately after -- consistent credit. |
| `9266e43f…` | Pass | No | 0.0169 | 0.0169 | 0.0000 | Pass into a `Ball Receipt*` landing in the same grid cell -- no net change, sensible for a short, low-risk pass. |
| `0974dcb2…` | Pass | No | 0.0102 | 0.0094 | +0.0007 | Small positive -- ball moves to a marginally safer cell. |
| `e37e4108…` | Pass | No | 0.0126 | 0.0143 | -0.0017 | Small negative -- possession continues into a marginally more dangerous zone, plausible mid-buildup. |
| `3ee67eff…` | Carry | No | 0.1081 | 0.4656 (shot xG) | **-0.3576** | Possession continues from a genuinely dangerous carry straight into a high-probability shot (xG 0.466) -- the corrected `xt_after` correctly reflects a big chance, something the 96-cell grid alone (max 0.2575) could never register on its own. |
| `d819f8ff…` | Carry | No | 0.0194 | 0.0238 (shot xG) | -0.0045 | Continues into a low-probability shot -- small, sensible negative delta. |
| `8fd13966…` | Pass | Yes | **0.2575** | 0.0000 | **+0.2575** | The single largest positive delta in the dataset -- a pass that happens to end possession from the single highest-value grid cell (deep, central, near the attacking goal) -- the most extreme "attack broke down at the worst possible moment for the attacking side" case, correctly credited at the grid's own maximum. |

### 2.5 Shared-target property, reconfirmed for the new target

Confirmed directly (section 0): `target_future_shot_10s`/`target_future_xg_10s` already show 0
events with more than one distinct value across the defender rows sharing an `event_id`.
`target_xt_delta_passive` is computed once per `event_id` by construction (section 1) and is meant to
be left-joined the same way — **the same value will be repeated identically across every defender row
sharing an `event_id`, exactly as the two existing targets already work.** No per-defender attribution
scheme was built or needed; that is explicitly out of scope for this prototype.

## 3. Honest conclusion

- **Schema and infrastructure transferred cleanly across the whole leg, no fallback needed anywhere**
  (section 0) -- the possession-outcome definition, built once for active, applies to passive without
  modification to its own logic, only to which columns it reads from.
- **The possession-ending collapse behaves exactly as designed**, mirroring active v2's own validated
  property.
- **The distribution is workable but not close to either prior target's shape**, and is *more*
  heavy-tailed than active v2's own already-heavy-tailed version (kurtosis 31 vs 18) -- the same
  hurdle-architecture concern Prompt 64/66 raised for active applies here too, arguably more acutely.
- **Correlation with the old targets improved in the right direction but is weaker than active's own
  improvement** (-0.06/-0.07 here vs -0.108 there) -- a real, reported difference, not smoothed over.
- Nothing here decides an EDA-suite build or a model-ladder rollout for this leg -- both remain
  explicitly out of scope for this prototype.
