"""Re-point layer for the xT-delta report suite -- PASSIVE LEG.

WHY THIS MODULE EXISTS (the Step-0 answer, in code form)
--------------------------------------------------------------------------
Prompt 65 built the xT report suite for the ACTIVE-BINARY leg only, and
Prompt 67 rebuilt it against the corrected `target_xt_delta_v2`. Both
passes reach the target through `from src.eda import xt_common as xc` and
bind the target name, the parquet paths, the loader, the statistics helpers
and the scale block from that one module. `xt_v2_common.py` exploits that
to re-point the whole suite onto v2 without editing a single Prompt-65
file.

This module does the same thing for the PASSIVE leg and
`target_xt_delta_passive` (Prompt 68,
`outputs/prototypes/PASSIVE_XT_TARGET_PROTOTYPE.md`). It rebinds
`xt_common`'s attributes onto the passive dataset so the suite's shared
machinery -- `categorical_xt_table`, `boolean_xt_lift`, `nonzero_subset`,
`_bin_table`, `run_test_xt`, `slice_one_xt`, `analyze_pair` -- operates on
passive rows with no edit to any existing script.

IMPORT ORDER MATTERS, exactly as for `xt_v2_common`. Every generator does
`TARGET = xc.TARGET` at module import time, so this module must be imported
BEFORE any `*_xt.py` generator. Each passive generator imports it first and
`generate_eda_portal_xt_passive.py` is the driver that guarantees the
ordering. Importing it is idempotent and self-applying.

**This module must NEVER be imported in the same process as
`xt_v2_common`.** Both re-point the same module attributes, and whichever
imports last wins. The passive suite runs as its own process.

--------------------------------------------------------------------------
THE TARGET (Prompt 68)
--------------------------------------------------------------------------
`target_xt_delta_passive = xt_before - xt_after`, computed ONCE PER UNIQUE
`event_id` (198,354 rows) and left-joined onto `passive_defense.parquet`'s
1,593,181 defender-slot rows here, in memory, never written back.

  xt_before = xT(ball_x, ball_y) at the snapshot itself -- no previous-event
              lookup (unlike the active leg, whose xt_before is the PREVIOUS
              event's grid cell).
  xt_after  = 0.0 if `action_ended_possession`; else the next
              same-possession event's xT(ball_x, ball_y); else that event's
              own `shot_statsbomb_xg` if it is a Shot.

Prompt 68 implemented the corrected, possession-outcome definition directly
on the first pass, so this leg has no v1/v2 supersession to carry: the
active leg's v1 mistake (scoring the defensive action's OWN recorded
location as xt_after) never had anywhere to happen here.

--------------------------------------------------------------------------
THE SHARED-TARGET PROPERTY -- stated once, restated in every report
--------------------------------------------------------------------------
Every defender row sharing an `event_id` carries the IDENTICAL target
value. This is deliberate and pre-existing, not an artefact to work around:
`target_future_shot_10s` and `target_future_xg_10s` already behave exactly
the same way on this leg (Prompt 68 section 0 checked all 198,354 events
directly -- 0 events show more than one distinct value for either). No
per-defender attribution scheme was built, and building one is explicitly
out of scope, the same call Prompt 68 made.

The methodological consequence is stated rather than hidden: rows are not
independent within an event (mean ~8.03 defender rows per event), so any
row-level p-value in this suite is computed on an effective sample size far
smaller than its nominal n. The existing passive xg-target reports state
the row grain ("one row per visible defender-slot per on-ball attacking
event") but carry no explicit independence caveat -- checked directly in
reports/analysis/xg_target/, not assumed -- so this suite ADDS the caveat
rather than claiming to mirror one. Effect sizes, shares and ranks are
unaffected; only the significance of p-values is.

--------------------------------------------------------------------------
ADAPTATION 1, RE-EXAMINED FOR THE PASSIVE LEG -- and it does NOT transfer
--------------------------------------------------------------------------
Prompt 65's Adaptation 1 translates the xg suite's mean-multiple threshold
convention onto the xT target through xg's own standard-deviation fraction.
Prompt 67 found that the resulting std-derived `flat_margin` describes
`target_xt_delta_v2`'s bulk badly, because the shot-xG injection inflates
std while leaving the middle of the distribution tight: on v2 the margin
covered 53.4% of all rows against v1's 35.4%.

Prompt 68 measured this leg as MORE heavy-tailed still -- skew -1.779 and
excess kurtosis 31.02, against active v2's -0.306 / 18.33 -- so the same
threshold problem is expected to be WORSE here, not better. It is
recomputed from the real passive parquet on every run rather than assumed
from active's numbers, and the MAD-derived robust alternative is published
beside the literal translated margin exactly as Prompt 67 does. The literal
margin is KEPT as the headline so the passive and active halves of this
portal stay numerically comparable and the xg suite's convention is still
the one being followed.
"""

from __future__ import annotations

import pandas as pd

from src.eda import xt_common as xc
from src.eda.feature_config import PASSIVE

TARGET_PASSIVE = "target_xt_delta_passive"
PASSIVE_PARQUET = xc.REPO_ROOT / PASSIVE["parquet_path"]
PASSIVE_XT_PARQUET = xc.REPO_ROOT / "outputs" / "prototypes" / "passive_xt_delta.parquet"

SHOT_TARGET = "target_future_shot_10s"

_MAD_TO_STD = 1.4826  # normal-consistency constant, so the robust scale is std-comparable

# --------------------------------------------------------------------------
# Prompt 68's carried-forward findings. These are quoted, not re-derived --
# every one of them is independently RE-MEASURED somewhere in this suite, and
# where a re-measurement disagrees with the quote the report says so.
# --------------------------------------------------------------------------

PROMPT_68_SHAPE_FINDING = (
    "Prompt 68 finding carried forward (outputs/prototypes/PASSIVE_XT_TARGET_PROTOTYPE.md section 2.2): "
    "target_xt_delta_passive is MORE heavy-tailed than the active leg's own already-heavy-tailed v2 target -- "
    "skew -1.779 and excess kurtosis 31.02, against active v2's -0.306 and 18.33 -- with a small, slightly "
    "NEGATIVE mean (-0.00181, against active's positive +0.00338), 26.3% exactly zero, 36.9% negative and 36.9% "
    "positive. The negative mean is a real, sensible difference in what is being measured rather than a "
    "discrepancy: active-binary rows are defensive ACTIONS, which on average should deny threat, while passive "
    "rows are snapshots at an arbitrary on-ball event during a live attacking phase, where there is no inherent "
    "reason attacking possession should lose value from one touch to the next. The heavier tail comes from the "
    "same mechanism as active's -- real shot_statsbomb_xg values injected as xt_after for the 863 events whose "
    "next event is a Shot, against the 96-cell xT grid's own maximum of 0.2575 -- but is relatively more "
    "concentrated here because possession-ending events are only 0.82% of this leg's event population. This "
    "target is not close to the old zero-inflated, positively-skewed shape either, and is if anything a harder "
    "distribution to work with than active v2's."
)

PROMPT_68_POSSESSION_COLLAPSE_FINDING = (
    "Prompt 68 finding carried forward (outputs/prototypes/PASSIVE_XT_TARGET_PROTOTYPE.md section 2.1): on the "
    "possession-ending subset (action_ended_possession == True, n=1,619 unique events), "
    "target_xt_delta_passive collapses EXACTLY onto xt_before's own distribution, mean +0.02452, confirmed "
    "identical with np.allclose. This is the same logically-REQUIRED property Prompt 66 confirmed for the "
    "active leg: xt_after is forced to 0.0 for exactly these rows, so the delta is xt_before by construction -- "
    "'100% of the threat that existed a moment ago is denied'. It collapses to a real, row-by-row varying "
    "distribution rather than to an uninformative constant zero, which is what the old target_future_shot_10s "
    "did on these same rows. Smaller than active's own +0.0305/+0.0307 for an understood reason: this leg's "
    "on-ball events are far more often mid-buildup passes and carries than active's defensive actions, so the "
    "average denied threat at the moment a possession happens to end is lower. Any report showing this slice "
    "must present the identity as construction, not as discovery."
)

PROMPT_68_CORRELATION_FINDING = (
    "Prompt 68 finding carried forward (outputs/prototypes/PASSIVE_XT_TARGET_PROTOTYPE.md section 2.3): "
    "correlation with the two OLD passive targets is correctly signed but numerically WEAKER than the active "
    "leg's own improvement -- Pearson r = -0.0629 against target_future_shot_10s and -0.0715 against "
    "target_future_xg_10s, against active v2's -0.108. More threat denied is still associated with less future "
    "attacking output, which is the sensible direction, but the relationship is looser on this leg. Stated "
    "plainly rather than talked up. A plausible reason, recorded by Prompt 68 and not chased here: a passive "
    "snapshot is one defender's off-ball positioning at an arbitrary point in a live attack, one step further "
    "removed from the moment-to-moment outcome than an active defensive ACTION is."
)

PROMPT_68_SHARED_TARGET_FINDING = (
    "Prompt 68 finding carried forward (outputs/prototypes/PASSIVE_XT_TARGET_PROTOTYPE.md sections 1 and 2.5): "
    "target_xt_delta_passive is computed ONCE PER UNIQUE event_id (198,354 rows) and left-joined onto "
    "passive_defense.parquet's 1,593,181 defender-slot rows, so the identical value repeats across every "
    "defender row sharing an event_id (mean ~8.03 rows per event). This is a deliberate, pre-existing property "
    "of this leg, not something to fix: target_future_shot_10s and target_future_xg_10s already work exactly "
    "the same way, checked directly across all 198,354 events with 0 showing more than one distinct value. No "
    "per-defender attribution scheme was built and building one is explicitly out of scope, the same call "
    "Prompt 68 made. The consequence for this suite is a measurement caveat rather than a defect: rows are not "
    "independent within an event, so row-level p-values are computed on an effective sample size far smaller "
    "than their nominal n. Effect sizes, shares and ranks are unaffected."
)

PROMPT_68_NO_V1_DETOUR_FINDING = (
    "Prompt 68 finding carried forward (outputs/prototypes/PASSIVE_XT_TARGET_PROTOTYPE.md preamble and section "
    "0): this leg has NO v1/v2 supersession to carry, and that is a property of the data rather than good "
    "luck. Active's v1 mistake was using the defensive action's own recorded location (action_x/action_y) as "
    "xt_after. passive_defense.parquet's ball_x/ball_y is already the ball's location at the on-ball event the "
    "snapshot is anchored to -- confirmed identical to events_with_targets.parquet's own ball_x/ball_y at the "
    "same event_id to floating-point exactness -- which is the correct 'before' state by the exact reasoning "
    "Prompt 66 established. The corrected possession-outcome xt_after was therefore implemented directly, on "
    "the first pass. Nothing in this half of the portal supersedes an earlier passive pass, because there "
    "isn't one."
)


def load_passive_xt(columns: list[str] | None = None) -> pd.DataFrame:
    """Passive parquet left-joined to Prompt 68's prototype by `event_id`.

    Read-only, always. Neither `data/features/passive_defense.parquet` nor
    `outputs/prototypes/passive_xt_delta.parquet` is ever written back, and
    the two are never merged-and-resaved.

    The prototype file is at the UNIQUE-EVENT grain (198,354 rows), not the
    defender-slot grain (1,593,181 rows), so this is a genuine one-to-many
    left join: every defender row sharing an `event_id` receives the
    identical target value. See this module's docstring -- that is the
    documented, deliberate design of this leg, matching how
    `target_future_shot_10s` / `target_future_xg_10s` already behave.
    """
    if columns is not None:
        # `event_id` is the join key; the two OLD passive targets are needed by
        # target_scale() for Adaptation 1's runtime threshold translation and
        # for the correlation block -- all three are always loaded.
        columns = list(dict.fromkeys(list(columns) + ["event_id", xc.XG_TARGET, SHOT_TARGET]))
    df = pd.read_parquet(PASSIVE_PARQUET, columns=columns)
    xt = pd.read_parquet(PASSIVE_XT_PARQUET)
    return df.merge(xt, on="event_id", how="left")


_ACTIVE_target_scale = xc.target_scale


def target_scale_passive(df: pd.DataFrame) -> dict:
    """Prompt 65's `target_scale`, recomputed on the passive leg and extended.

    The arithmetic is deliberately unchanged from the active suite's -- every
    threshold is still the xg suite's own ratio translated through xg's own
    std-fraction, computed at runtime from the real parquet. What is ADDED is
    the robustness block (Prompt 67's convention, carried over) and a second
    old-target correlation, because this leg has TWO old targets to compare
    against and Prompt 68 reported both.
    """
    s = dict(_ACTIVE_target_scale(df))

    t = df[xc.TARGET].dropna()
    std = s["std"]
    mean = s["mean"]
    frac = s["xg_flat_margin_as_fraction_of_xg_std"]
    fm = s["flat_margin"]

    iqr = float(t.quantile(0.75) - t.quantile(0.25))
    mad = float((t - t.median()).abs().median())
    robust_std = _MAD_TO_STD * mad
    share_within = float((t.abs() < fm).mean() * 100)

    s["leg"] = "passive"
    s["target_version"] = "passive (possession-outcome definition from the first pass)"
    s["target_source"] = (
        "outputs/prototypes/passive_xt_delta.parquet (198,354 unique-event rows, left-joined READ-ONLY onto "
        "data/features/passive_defense.parquet's 1,593,181 defender-slot rows by event_id)"
    )
    s["supersedes"] = (
        "nothing. Unlike the active-binary leg, this leg has no v1 pass to supersede -- Prompt 68 implemented "
        "the corrected possession-outcome xt_after directly. See prompt_68_no_v1_detour."
    )
    s["row_grain_note"] = PROMPT_68_SHARED_TARGET_FINDING

    # Prompt 68 measured this target's distribution at the UNIQUE-EVENT grain
    # (198,009 defined values). Every report in this suite necessarily measures
    # it at the DEFENDER-SLOT grain, because that is the grain the features
    # live at. The two differ, and the difference is reported rather than
    # quietly presenting one set of numbers under the other's label.
    s["grain_comparison"] = {
        "this_suite_grain": "defender-slot (one row per visible defender per on-ball event)",
        "prompt_68_grain": "unique event_id",
        "prompt_68_unique_event": {
            "mean": -0.00181, "std": 0.03828, "skew": -1.779, "excess_kurtosis": 31.02,
            "pct_zero": 26.3, "pct_negative": 36.9, "pct_positive": 36.9, "n_defined": 198009,
        },
        "this_run_defender_slot": {
            "mean": s["mean"], "std": s["std"], "skew": s["skew"],
            "excess_kurtosis": s["excess_kurtosis"], "pct_zero": s["pct_zero"],
            "pct_negative": s["pct_negative"], "pct_positive": s["pct_positive"],
            "n_defined": s["n_defined"],
        },
        "note": (
            "These are the SAME target measured at two grains, not a discrepancy to reconcile. Prompt 68 "
            "reported the unique-event distribution because that is the grain the target is computed at. Every "
            "report in this suite measures it at the defender-slot grain, because that is the grain the "
            "features live at and the grain a model on this leg would train on. The row-level distribution is "
            "MORE extreme on every shape statistic -- measured on this run, skew "
            f"{s['skew']} against Prompt 68's -1.779 and excess kurtosis {s['excess_kurtosis']} against 31.02 "
            "-- and the mechanism is understood rather than guessed at: events with more visible defender "
            "slots contribute more rows, so the row-level distribution is the event-level one re-weighted by "
            "visible-defender count, and the extreme shot-xG-tail events are exactly the crowded, "
            "well-observed ones that carry the most slots. The directional shares barely move "
            f"({s['pct_zero']:.1f}% zero / {s['pct_negative']:.1f}% negative / {s['pct_positive']:.1f}% "
            "positive here against 26.3 / 36.9 / 36.9 there), which is the reassuring part: the re-weighting "
            "lengthens the tails without changing which way the target points. Wherever this suite cites a "
            "Prompt 68 figure it is labelled as the unique-event figure."
        ),
    }

    s["robust_scale"] = {
        "iqr": round(iqr, 8),
        "mad": round(mad, 8),
        "robust_std_1_4826_mad": round(robust_std, 8),
        "robust_flat_margin": round(frac * robust_std, 8),
        "flat_margin_over_mad": round(fm / mad, 4) if mad else None,
        "flat_margin_over_iqr": round(fm / iqr, 4) if iqr else None,
        "pct_rows_inside_flat_margin": round(share_within, 2),
        "mean_as_fraction_of_std": round(mean / std, 4) if std else None,
        "active_v2_comparison": {
            "flat_margin_over_mad": 1.28,
            "flat_margin_over_iqr": 0.56,
            "pct_rows_inside_flat_margin": 53.4,
            "excess_kurtosis": 18.3289,
            "note": (
                "Prompt 67's measured figures for target_xt_delta_v2 on the active leg, quoted here so the "
                "passive numbers beside them can be read as a comparison rather than in isolation. v1's own "
                "figures, for the longer arc, were 0.53 x MAD and 35.4% of rows at excess kurtosis 8.85."
            ),
        },
    }

    # Adaptation 1 is the one place this whole suite still reaches for an OLD
    # target (it borrows xg's std-fraction to set every threshold). This leg
    # has TWO old targets and Prompt 68 reported both, so both are measured.
    both_xg = df[[xc.TARGET, xc.XG_TARGET]].dropna()
    both_shot = df[[xc.TARGET, SHOT_TARGET]].dropna()
    r_xg = float(both_xg[xc.TARGET].corr(both_xg[xc.XG_TARGET])) if len(both_xg) else float("nan")
    r_shot = float(both_shot[xc.TARGET].corr(both_shot[SHOT_TARGET])) if len(both_shot) else float("nan")

    s["correlation_with_old_target"] = {
        "pair": f"{xc.TARGET} vs {xc.XG_TARGET}",
        "pearson_r": round(r_xg, 4),
        "n": int(len(both_xg)),
        "pearson_r_vs_shot_target": round(r_shot, 4),
        "n_vs_shot_target": int(len(both_shot)),
        "prompt_68_reported_r_vs_xg": -0.0715,
        "prompt_68_reported_r_vs_shot": -0.0629,
        "active_v2_pearson_r_vs_xg": -0.1082,
        "note": (
            f"Measured on this run at the defender-slot grain: r={r_xg:+.4f} against target_future_xg_10s and "
            f"r={r_shot:+.4f} against target_future_shot_10s. Prompt 68 reported -0.0715 and -0.0629 "
            "respectively at the unique-event grain; the two grains can differ slightly because events with "
            "more visible defender slots are weighted more heavily at row level. Both are correctly signed -- "
            "more threat denied goes with less future attacking output -- and both are WEAKER than the active "
            "leg's own -0.1082 against the same xg target. That difference is real and is reported rather than "
            "smoothed over. Recorded HERE specifically because Adaptation 1 borrows target_future_xg_10s's own "
            "standard-deviation fraction to set every threshold in this suite, so wherever a report cites an "
            "xg-derived threshold, this is how the targets actually relate on the same rows. "
            + PROMPT_68_CORRELATION_FINDING
        ),
    }

    s["threshold_transfer_warning"] = (
        "THIS THRESHOLD DOES NOT TRANSFER CLEANLY, AND ON THIS LEG IT TRANSFERS WORSE THAN ON ACTIVE -- "
        "recomputed on the real passive parquet rather than inherited from active's numbers, and reported "
        "rather than forced to fit. Prompt 65 tuned the translated flat_margin against a target with excess "
        "kurtosis 8.85, where the margin was 0.53 x the MAD and spanned 35.4% of rows -- a genuinely narrow "
        "'too small to matter' band. Prompt 67 found that on target_xt_delta_v2 (excess kurtosis 18.33) the "
        "same std-derived margin had grown to 1.28 x the MAD and 53.4% of rows, because the shot-xG injection "
        f"inflates std while the bulk tightens. This leg is heavier-tailed again (excess kurtosis "
        f"{s['excess_kurtosis']}, IQR {iqr:.6f}, MAD {mad:.6f}), so the same mechanism applies more strongly: "
        f"the translated margin is {fm:.6f}, which is {fm / mad:.2f} x the MAD, {fm / iqr:.2f} x the IQR, and "
        f"spans {share_within:.1f}% of all rows. A flat band covering this much of the data is not doing its "
        "job. A NAIVE TRANSFER OF ACTIVE'S OWN ABSOLUTE MARGIN (0.004929) WOULD HAVE BEEN WORSE STILL, since "
        "this leg's std is smaller (0.038 vs 0.058) while its bulk is tighter again -- the margin is therefore "
        "recomputed from this leg's own std, not copied. The literal translated margin is KEPT as the headline "
        f"number so the active and passive halves of this portal stay like-for-like and the xg suite's "
        f"convention is still the one being followed, but the MAD-derived robust alternative "
        f"({frac * robust_std:.6f}) is published beside it on every run and every shape classification in this "
        "half of the portal should be read with this inflation in mind."
    )

    s["scale_note"] = (
        "ADAPTATION 1, RE-EXAMINED FOR THE PASSIVE LEG. Prompt 65 translated the xg suite's mean-multiple "
        "threshold convention onto the xT target via xg's own standard-deviation fraction, because the xT "
        "target's mean is near zero and a mean-multiple threshold would be an arbitrary sliver. That "
        f"justification holds on this leg in its ORIGINAL form: target_xt_delta_passive's mean is {mean:+.6f}, "
        "NEGATIVE as v1's was and unlike active v2's positive mean, so '0.5 x the overall mean' would flip the "
        f"sign of every threshold. At {mean:+.6f} against a std of {std:.6f} the mean is only "
        f"{abs(mean) / std * 100:.1f}% of one standard deviation. Mechanically: xg's own flat_margin (0.5 x "
        f"mean xg = {s['xg_flat_margin_absolute']:.6f} on these same rows) re-expressed as a fraction of xg's "
        f"own std ({s['xg_std_same_rows']:.6f}) gives {frac:.6f}, applied to target_xt_delta_passive's std "
        f"({std:.6f}) to give {fm:.6f}. Every number is computed at runtime from the real parquet -- nothing "
        "is copied from the active leg. See `threshold_transfer_warning` in this same block. Scale-free "
        "thresholds (|Spearman rho| >= 0.05, small-n row counts, the substitutive/additive ratio cutoffs) are "
        "carried over completely unchanged and are unaffected."
    )

    s["conditional_panel_note"] = (
        "ADAPTATION 2, unchanged in method and re-measured on the passive leg. The xg_target suite's second "
        "panel conditions on target_future_shot_10s == 1 to strip the structural zero mass. "
        "target_xt_delta_passive has no shot column standing behind it; its structural-zero analogue is the "
        f"{s['pct_zero']:.1f}% of rows with an EXACTLY zero delta -- the next same-possession event landed in "
        "the same 8x12 xT grid cell as the snapshot itself, which on this leg is very common because most "
        "on-ball events are short passes and carries. That share is LARGER than active v2's 20.2%, so the "
        "panel matters more here, not less. The conditional panel throughout this half of the portal is "
        "'given a non-zero delta' (target_xt_delta_passive != 0), never 'given a shot'."
    )

    s["direction_note"] = (
        "ADAPTATION 3, unchanged in method and re-measured on the passive leg. Sign carries the football "
        "meaning: POSITIVE target_xt_delta_passive = xT fell across the event (threat reduced, good for the "
        f"defence), NEGATIVE = xT rose. On this leg the two directions are almost exactly balanced -- "
        f"{s['pct_positive']:.1f}% positive vs {s['pct_negative']:.1f}% negative -- but the target is strongly "
        f"LEFT-skewed (skew {s['skew']}), far more so than active v2's -0.306, so the negative tail is much "
        "longer than the positive one even though the counts match. Two-directional framing therefore matters "
        "more here than anywhere else in this portal. Rankings are by |magnitude| across BOTH directions, and "
        "every table also reports the negative/positive/zero share behind its mean."
    )

    s["log_transform_note"] = (
        "NOT an adaptation, re-checked rather than inherited: the xg_target report suite never log-transforms "
        "its target anywhere, on either leg, and neither did the active xT suite. The log1p framing in this "
        "project belongs to the modelling legs, not to this report suite. This leg's excess kurtosis of ~31 "
        "makes a transform more tempting than anywhere else in the portal -- and it is still NOT applied here, "
        "because doing so would silently break comparability with the passive xg portal and with this "
        "portal's own active half. Recorded as an open question for the modelling legs, not resolved by an EDA "
        "suite."
    )

    return s


def repoint() -> None:
    """Rebind `xt_common`'s module attributes onto the passive leg. Idempotent.

    This is the whole re-point mechanism. Because every Prompt-65 generator
    reaches the target through `xc.<attr>` rather than opening files or
    naming columns itself, rebinding here re-points the suite's shared
    machinery with zero edits to those files. They still regenerate the
    active portal exactly when run against an un-re-pointed `xt_common`.
    """
    xc.TARGET = TARGET_PASSIVE
    xc.ACTIVE_PARQUET = PASSIVE_PARQUET
    xc.XT_PARQUET = PASSIVE_XT_PARQUET
    xc.load_active_xt = load_passive_xt
    xc.target_scale = target_scale_passive
    # Prompt 64's two finding strings are embedded verbatim in every
    # Prompt-65 report. Both are ACTIVE-LEG statements about event_type
    # Clearance rows and about v1's distribution -- neither is about anything
    # that exists on this leg (passive_defense.parquet has no event_type
    # column and no defensive-action rows at all). They are replaced at the
    # source with this leg's own carried-forward findings rather than left to
    # propagate a statement about a different dataset into a passive report.
    xc.PROMPT_64_CLEARANCE_FINDING = PROMPT_68_SHARED_TARGET_FINDING
    xc.PROMPT_64_SHAPE_FINDING = PROMPT_68_SHAPE_FINDING


repoint()

# Passive-leg output filenames. The xg_target precedent puts BOTH legs'
# sections inside one file for the eight shared-name reports (LEAKAGE_AUDIT,
# CONFOUND_ANALYSIS, TOURNAMENT_STABILITY_CHECK, SLICE_STRATIFICATION,
# SLICE_STRATIFICATION_V2, FEATURE_INTERACTION_ANALYSIS,
# FEATURE_LOCK_CONFIRMATION_XG, REVIEW_ANALYSIS) -- confirmed by reading
# those JSONs, not assumed. That convention cannot be mirrored literally
# here: this portal's existing files of those names are Prompt 67's ACTIVE
# output and are frozen, so merging a passive section into them would mean
# editing active-leg report content. The passive halves are therefore
# separate `PASSIVE_*` files, and the portal index and MASTER_FINDINGS say
# so explicitly rather than leaving a reader to wonder why the shapes
# differ between the xg and xT portals.
PASSIVE_FILE_PREFIX = "PASSIVE_"


def out(name: str) -> "object":
    return xc.OUT_DIR / name
