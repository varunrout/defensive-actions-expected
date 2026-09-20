# xT-Delta Target Prototype V2 — Corrected `xt_after` (Active-Binary Leg Only)

Supersedes `XT_TARGET_PROTOTYPE.md` (Prompt 64) for the `xt_after` definition only. Prototype,
active-binary leg only — the passive leg is out of scope, and the Prompt 65 EDA report suite
(`reports/analysis/xt_target/`) is **not** rebuilt here; it is now known-stale against this
corrected target and gets its own follow-up prompt. `player_defensive_actions.parquet`,
`short_horizon.py`, and the v1 file (`outputs/prototypes/active_binary_xt_delta.parquet`) are all
read-only / untouched in this prompt.

## 0. Confirmed schema, before writing anything

**Chronological event order.** `src/dax/features/event_context.py`'s own docstring states the
canonical order plainly: `match_id, period, index` (not raw row order in the parquet — checked
directly and reconfirmed: `data/processed/events_with_targets.parquet`'s `index` column is globally
contiguous *within a match, across periods* — e.g. period 1 ends at index 1825, period 2 begins at
index 1826 with no gap or reset — but `event_context.py`'s own possession-flag derivation groups by
`["match_id", "period"]` before taking `.shift(-1)`, meaning **a "next event" lookup must respect
period boundaries** to stay consistent with how `action_ended_possession` was itself computed (a
half-time break is a real discontinuity, not a continuation). This prototype's "next event" =
same `match_id` **and** same `period`, `index + 1`.

**`action_ended_possession` vs `action_won_possession`.** Read `event_context.py`'s source directly
(not inferred from column names):
```
action_changed_possession = (next_possession_team != possession_team)  # via period-grouped shift(-1)
action_ended_possession    = action_changed_possession
action_won_possession      = action_changed_possession & actor_was_defending
```
**`action_ended_possession` is the correct, general "this row's possession ends here" flag** — it is
purely about whether the very next event (respecting period boundaries) belongs to a different team,
regardless of who performed this row's action. `action_won_possession` is a narrower subset (also
requires the actor to be on the defending side) — on the locked active-binary rows, 4,662 rows have
`action_ended_possession == True`, of which 3,605 also have `action_won_possession == True` (1,057
rows end their possession without being flagged as "won" by the acting defender specifically — a real
but expected gap given `event_semantics_known`/`actor_was_defending` requires clean team-context
resolution that not every row clears). **`action_ended_possession` drives the `xt_after = 0` rule**;
`action_won_possession` is checked separately in section 2 as a secondary validation slice, per the
task's own request, not as the rule's trigger.

**Better location field for the previous action's own end-point?** Checked: StatsBomb's schema only
populates `pass_end_location` / `carry_end_location` / `shot_end_location` / `goalkeeper_end_location`
for those specific attacking event types. None of the defensive event types this leg's rows are drawn
from (`Clearance`, `Interception`, `Block`, `Duel`, `Pressure`, `Ball Recovery`, `Foul Committed`,
`50/50`) carry their own populated end-location field. **No better field exists** — the next event's
own location genuinely is the right (and only) way to know "where did the ball actually go," matching
the task's own framing and the standard VAEP/xT convention (value the *next touch*, not the current
action's coordinates).

**Shot as the next event.** Checked the actual frequency first: of the 51,406 rows where possession
continues, **954 (1.9%) have a `Shot` as the immediately next event** — not a negligible edge case.
Decision, reasoned explicitly: value these via the shot's own `shot_statsbomb_xg` (populated for
100% of these 954 rows, checked directly — 0 NaN), **not** a grid lookup at the shot's location. This
matches the literature convention this whole approach is modelled on: `socceraction`'s own
`ExpectedThreat.rate()` explicitly excludes shots from its movement-value grid (`get_successful_move_actions`
filters to passes/dribbles/crosses only; shots are left `NaN` in that function) — a shot's value is a
scoring-probability question, not a "value of continuing possession from this zone" question, and
StatsBomb's own shot xG model is a strictly better instrument for that than a 96-cell grid. One real
scale consequence, worth flagging rather than smoothing over: shot xG for a clear chance can exceed
the xT grid's own maximum cell value (grid max 0.2575; **10.8% of all shots in this dataset have
`shot_statsbomb_xg > 0.2575`**) — meaning the largest `xt_after` values in this corrected target now
come from genuinely big chances conceded immediately after a defensive action, which is exactly the
signal a defensive-value target should carry, not an artifact to clip away.

**Period/match-end edge case (no next event, no flag).** Checked directly: **0 of 56,068 rows** are
the literal last event of their period with `action_ended_possession` still `False` — every row that
has no genuine next event within its period is already correctly flagged by
`action_ended_possession`. No separate fallback branch was needed in practice (the code still guards
for it defensively, in case a future data refresh changes this).

## 1. Implementation

`scripts/analysis/build_xt_delta_v2_prototype.py`. `xt_before` is **unchanged from Prompt 64** (same
method: previous event's `ball_x`/`ball_y`, looked up by `match_id` + `index - 1`, no period
restriction — that side of the definition was explicitly out of scope this prompt). `xt_after`:

- `action_ended_possession == True` (4,662 rows): `xt_after = 0.0`.
- Otherwise, next event is a `Shot` (954 rows): `xt_after = shot_statsbomb_xg` of that shot.
- Otherwise (50,452 rows): `xt_after = xT(next event's ball_x, ball_y)` via the unchanged
  `src/dax/targets/expected_threat.py` grid lookup.

`target_xt_delta_v2 = xt_before - xt_after`. Written to
`outputs/prototypes/active_binary_xt_delta_v2.parquet` (`event_id`, `xt_before`, `xt_after`,
`target_xt_delta_v2`), row-aligned by `event_id`, not merged into the locked parquet. 307 of 56,068
rows (0.55%) are `NaN` overall: 32 from `xt_before` (unchanged from v1 — previous event is a non-ball
event like `Tactical Shift`/`Substitution`) plus 277 from `xt_after` (next event has no populated
`ball_x`/`ball_y`, ~0.5% missing-location rate, same order of magnitude as v1's own missing rate) —
neither silently zero-filled.

## 2. Re-validation

### 2.1 The two possession-ending slices — now a deterministic, logically correct property

| Slice | n (defined) | `target_xt_delta_v2` mean | `xt_before` mean (same slice) | Identical? |
|---|---|---|---|---|
| `action_ended_possession == True` | 4,656 | **+0.03051** | +0.03051 | **Yes, exactly** (checked with `np.allclose`) |
| `action_won_possession == True` | 3,600 | **+0.03066** | +0.03066 | **Yes, exactly** |

This is **expected and logically correct, not a new degenerate finding**: since `xt_after` is forced
to `0.0` for every row in these slices, `target_xt_delta_v2` collapses to `xt_before` exactly — "100%
of the threat that existed a moment ago is denied," which is precisely what should be true when a
defensive action ends the attack. Unlike v1's old target (`target_future_shot_10s`, which collapsed
these rows to an *identical, uninformative* zero for every row regardless of how dangerous the
situation had been), this collapses to `xt_before`'s own **real, varying distribution** (mean
+0.031, std ~0.050, the actual danger level that was denied) — informative by construction, and
confirmed to genuinely vary row-by-row, not a repeated constant.

### 2.2 The Clearance artifact — reverses direction, does not persist

| | v1 (`action_x`-based `xt_after`) | v2 (possession-outcome `xt_after`) |
|---|---|---|
| Clearance, `action_ended_possession==True` slice (n=289) | **-0.0315** | **+0.0406** |
| Clearance, all rows (n=4,096-4,187) | **-0.0325** | **+0.0247** |
| Clearance's rank among event types by mean, all rows | **8th of 8 (worst)** | **1st of 8 (best)** |

**The artifact fully reverses, both in the motivating slice and overall — it does not persist in any
form.** Clearance goes from the single worst-looking event type under v1 to the single best-looking
under v2, a complete rank reversal (8th -> 1st), and the sign itself flips from clearly negative to
clearly positive. This is exactly the mechanism-level fix the task asked to verify rather than
assume: the root cause (using the clearance's own action location, deep in the box, as `xt_after`)
is gone under v2 because possession-ending clearances now get `xt_after = 0` by construction, not a
grid lookup at a dangerous coordinate at all.

*(Aside, not chased further here since it's outside this prompt's scope: `Block` is now the
lowest-mean event type (-0.0185) under v2, not `Clearance`. Worth a look in a future pass, not a
regression of anything this prompt was asked to fix.)*

### 2.3 Distribution stats, v1 vs v2

| Statistic | v1 (Prompt 64) | v2 (this prompt) |
|---|---|---|
| n (defined) | 56,036 | 55,761 |
| Mean | -0.00132 | **+0.00338** |
| Std | 0.06499 | 0.05838 |
| Skew | +0.093 (near-symmetric) | **-0.306** (mildly left-skewed) |
| Excess kurtosis | 8.85 | **18.33** (much heavier tails) |
| Share exactly zero | 11.3% | 20.2% |
| Share negative | 47.0% | 35.4% |
| Share positive | 41.7% | 44.4% |

**Mean flips from slightly negative to clearly positive** — a real, welcome change: on average,
active defensive actions now genuinely deny threat (+0.0034/row on average), which is the
football-sensible direction a defensive-value target should point in, and v1 never achieved this.
**Heavier tails and a left skew** are a direct, understood consequence of injecting `shot_statsbomb_xg`
(up to 0.897) as `xt_after` for the 954 shot-follows-defensive-action rows — those rows can produce
large negative deltas (a big chance conceded right after a defensive action is a genuinely bad
outcome, correctly reflected as a large negative value) that the coarse 96-cell grid alone could
never produce (its own max cell is only 0.2575). **This further reinforces Prompt 64's own flag that
the current hurdle architecture (built for a zero-inflated, positive-skewed target) does not apply
to this target as-is** — v2 is if anything less like that shape than v1 was, not more.

### 2.4 Correlations

| Pair | Pearson r |
|---|---|
| `target_xt_delta_v2` vs `target_future_xg_10s` (old target) | **-0.108** |
| `target_xt_delta_v2` vs `target_xt_delta` (v1) | **+0.571** |
| `target_xt_delta` (v1) vs `target_future_xg_10s` (recomputed for reference) | -0.016 |

**v2 shows a real, directionally sensible relationship with the old target that v1 never had.**
-0.108 is still modest, but it is the *right sign* and an order of magnitude stronger than v1's
essentially-zero -0.016: more threat denied by a defensive action (`target_xt_delta_v2` high) is
associated with less future attacking output in the next 10 seconds (`target_future_xg_10s` low) —
exactly the relationship a defensive-value target should show, even though the two targets are
measuring genuinely different things (instantaneous value-swing vs. forward-looking outcome) and are
not expected to be strongly correlated. **v2 and v1 are moderately correlated (+0.571)** — sensible,
since they share the exact same `xt_before` term and differ only in `xt_after`.

### 2.5 Spot-check, 10 rows, v1 vs v2 side by side

| `event_id` (short) | `event_type` | ended? | `xt_before` | `xt_after` v1 | `delta` v1 | `xt_after` v2 | `delta` v2 | Read |
|---|---|---|---|---|---|---|---|---|
| `0b10f597…` | Clearance | Yes | 0.0094 | 0.2575 | **-0.2480** | **0.0000** | **+0.0094** | v2 correctly credits a box clearance; v1 penalised it for the defender's own dangerous starting position. |
| `82317ad3…` | Clearance | Yes | 0.0169 | 0.0295 | -0.0127 | 0.0000 | **+0.0169** | Same pattern, smaller magnitude. |
| `4422e6b7…` | Clearance | Yes | 0.0089 | 0.2575 | -0.2486 | 0.0000 | **+0.0089** | Same pattern as row 1. |
| `a5ca59f6…` | Ball Recovery | Yes (won) | 0.0351 | 0.1081 | -0.0729 | 0.0000 | **+0.0351** | Possession-ending recovery, correctly credited with the full danger level denied. |
| `2b8078e0…` | Pressure | Yes (won) | 0.0143 | 0.0169 | -0.0026 | 0.0000 | **+0.0143** | Same pattern -- small magnitude, correct sign flip. |
| `3fd117c3…` | Pressure | No (continues) | 0.0644 | 0.0089 | +0.0555 | 0.0141 (shot xG) | +0.0504 | Possession continues into a Shot (xG 0.014, low-danger effort) -- v2 uses the shot's own xG, not a grid cell; v1's grid-based figure happened to be similar here by coincidence. |
| `49e75f5d…` | Ball Recovery | No (continues) | 0.0098 | 0.0100 | -0.0002 | 0.0277 (shot xG) | **-0.0179** | Possession continues into a Shot with meaningfully higher xG (0.028) than the grid cell v1 used (0.010) -- v2 correctly reflects that the danger a moment later was worse than the raw location would suggest. |
| `5500d603…` | Pressure | No (continues) | 0.0127 | 0.0194 | -0.0067 | 0.0127 (next event, non-shot) | 0.0000 | Ordinary continuing possession, next event lands in the same-value cell as `xt_before` under v2 -- delta correctly reads as no net change. |
| `34bacb71…` | Block | No (continues) | 0.0286 | 0.0108 | +0.0177 | 0.0094 (next event) | +0.0191 | Similar in both versions here -- v2's next-event lookup and v1's own-location lookup happen to land in nearby cells for this row. |
| `d3ffbec7…` | Foul Committed | No (continues) | 0.0240 | 0.0106 | +0.0134 | 0.0240 (next event) | 0.0000 | Foul restarts play at essentially the same threat level as before -- sensible; v1 had used the foul's own location and got a different (also plausible, but not comparably grounded) reading. |

## 3. Honest conclusion

- **The core correction works.** The Clearance artifact is not just reduced but fully reversed (8th
  -> 1st of 8 event types by mean), and the mechanism is understood, not assumed: possession-ending
  actions now collapse to `xt_before`'s own real distribution (mean +0.031) rather than either the
  old target's uninformative zero or v1's location-based distortion.
- **The distribution moved in the expected, football-sensible direction on the headline statistic**
  (mean flips from slightly negative to clearly positive) but got **more heavy-tailed and skewed, not
  less** — a real, understood side effect of correctly valuing shot outcomes via their own xG rather
  than a coarse grid cell. This reinforces, rather than resolves, Prompt 64's open flag that the
  current hurdle architecture does not fit this target's shape as-is.
- **Correlation with the old target improved from near-zero to a real, correctly-signed relationship**
  (-0.016 -> -0.108) — modest, but directionally exactly what a defensive-value target should show.
- Nothing here decides the Prompt 65 EDA-suite rebuild or a model-ladder rollout — both remain
  explicitly out of scope for this prompt, per the task brief.
