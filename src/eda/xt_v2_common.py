"""Re-point layer for the xT-delta v2 report suite -- ACTIVE-BINARY LEG ONLY.

WHY THIS MODULE EXISTS (the Step-0 answer, in code form)
--------------------------------------------------------------------------
Prompt 65's 21 `*_xt.py` generators were written against `target_xt_delta`
(v1). Prompt 66 corrected the target's `xt_after` term and produced
`target_xt_delta_v2`. Re-reading all 21 scripts showed that they share one
very fortunate property: **every single one of them reaches the target
through `from src.eda import xt_common as xc`**, and binds the target name,
the parquet path, the output directory, the statistics helpers and the
carried-forward finding strings from that one module. None of them opens
the prototype parquet directly and none of them hardcodes the target column
name in a computation.

That means the great majority of the suite is re-pointable by **rebinding
`xt_common`'s module attributes before the generators are imported**, with
no edit to any of Prompt 65's files. Those 21 files therefore stay exactly
as they were and still regenerate the v1 portal (now preserved at
`reports/analysis/xt_target_v1_superseded/`) byte-for-byte if re-run against
an un-re-pointed `xt_common`.

Where a script bakes in a v1-specific *scale or direction assumption* --
prose asserting a near-zero NEGATIVE mean, a "Clearance is the worst event
type" note, a confound test whose whole premise was the v1 artefact -- a
re-point is NOT enough and a new v2 sibling generator exists instead. Those
are listed in the Step-0 table in the rebuilt portal's MASTER_FINDINGS.

IMPORT ORDER MATTERS. Every generator does `TARGET = xc.TARGET` at module
import time, so this module must be imported BEFORE any `*_xt.py`
generator. `generate_eda_portal_xt_v2.py` is the driver that guarantees
that ordering; importing this module is idempotent and self-applying.

--------------------------------------------------------------------------
WHAT CHANGED IN THE TARGET (Prompt 66)
--------------------------------------------------------------------------
v1:  xt_after = xT(action_x, action_y)   -- the defensive action's OWN
     recorded location. For a Clearance that is where the ball WAS when
     cleared (often deep in the defending box, a high-xT cell), not where
     it ended up.
v2:  xt_after = 0.0 if the action ends its own possession
     (`action_ended_possession == True`, 4,662 rows);
     else `shot_statsbomb_xg` of the next event if that event is a Shot
     (954 rows, 1.9% of continuing-possession rows);
     else xT(next same-possession event's ball_x, ball_y).

`xt_before` is UNCHANGED between v1 and v2.

--------------------------------------------------------------------------
ADAPTATION 1, RE-EXAMINED AGAINST v2 -- and it does NOT transfer cleanly
--------------------------------------------------------------------------
Prompt 65's Adaptation 1 translated the xg suite's mean-multiple threshold
convention onto the xT target via xg's own std-fraction, because v1's mean
(-0.0013) was near zero and NEGATIVE, so "0.5 x the mean" was meaningless.

Two things changed, and they are reported rather than smoothed over:

1. v2's mean is now POSITIVE (+0.00338). The original *reason* for the
   translation (a negative mean flips the sign of every threshold) no
   longer applies in its literal form. The translation is nonetheless KEPT,
   for a reason that does still hold and is stated in every report: at
   +0.00338 against a std of 0.0584, the mean is still only 5.8% of a
   standard deviation, so a mean-multiple threshold would be an essentially
   arbitrary sliver -- and keeping the translation is what makes the v1 and
   v2 portals numerically comparable at all.

2. **The std-based translation itself does not transfer cleanly to v2, and
   this is the single threshold in the suite that breaks.** v2 is far more
   heavy-tailed than v1 (excess kurtosis 8.85 -> 18.33), because Prompt 66
   injects real `shot_statsbomb_xg` values up to 0.897 for the 954 rows
   where a Shot follows the defensive action -- values the 96-cell xT grid
   (max cell 0.2575) could never produce. Those tails inflate the standard
   deviation while the *bulk* of the distribution got much tighter:

       v1:  std 0.064986   IQR 0.019119   MAD 0.010277
       v2:  std 0.058385   IQR 0.008829   MAD 0.003853

   The translated flat margin barely moves (0.005486 -> 0.004929, -10%)
   because it is std-derived, but the core of the distribution more than
   halved. The consequence is concrete and measurable:

       v1:  flat_margin = 0.53 x MAD, 0.29 x IQR, spans 35.4% of rows
       v2:  flat_margin = 1.28 x MAD, 0.56 x IQR, spans 53.4% of rows

   A "this difference is too small to matter" band that covers the majority
   of the data is not doing the job it was designed for. `flat_margin`
   feeds `classify_shape()` in five reports, so this directly inflates how
   many binned curves get called "flat" on v2 relative to v1.

   DECISION, stated plainly rather than papered over: the literal translated
   `flat_margin` is KEPT as the headline threshold, so v1-vs-v2 comparisons
   stay like-for-like and the xg suite's convention is still the one being
   followed. But a robust, MAD-derived alternative is computed alongside it
   on every run, the ratios above are published in every report's
   `target_scale` block, and every report that uses `flat_margin` for a
   shape classification carries the warning. Silently swapping in the
   robust threshold would have made every v2 number incomparable with the
   v1 portal it supersedes; silently reusing the std one without saying
   that it now swallows half the data would have been worse.
"""

from __future__ import annotations

from src.eda import xt_common as xc

TARGET_V2 = "target_xt_delta_v2"
XT_PARQUET_V2 = xc.REPO_ROOT / "outputs" / "prototypes" / "active_binary_xt_delta_v2.parquet"
V1_PORTAL_DIRNAME = "xt_target_v1_superseded"

# --------------------------------------------------------------------------
# Prompt 66's carried-forward findings -- these REPLACE Prompt 64's, which
# Prompt 65 embedded in every report. Prompt 64's Clearance finding is now
# known to have been a measurement artefact that v2 REMOVED, so repeating it
# unchanged in a v2 report would assert something false.
# --------------------------------------------------------------------------

PROMPT_66_CLEARANCE_FINDING = (
    "Prompt 66 finding carried forward (outputs/prototypes/XT_TARGET_PROTOTYPE_V2.md section 2.2): Prompt 64's "
    "Clearance artefact -- Clearance looking like the WORST event type because xt_after was looked up at the "
    "clearance's own deep-in-the-box location -- is not merely reduced under the corrected xt_after, it FULLY "
    "REVERSES. Clearance moves from 8th of 8 event types (worst) at -0.0325 mean delta under v1 to 1st of 8 "
    "(best) at +0.0247 under v2, and in the motivating action_ended_possession slice from -0.0315 to +0.0406. "
    "The mechanism is understood, not assumed: possession-ending clearances now get xt_after = 0 by "
    "construction, so the dangerous coordinate is never looked up at all. Any v1-era statement that Clearance "
    "ranks last, or that clearances carry a systematically negative delta, is superseded and must not be "
    "repeated. (Aside recorded by Prompt 66 and not chased here: Block, at -0.0185, is now the lowest-mean "
    "event type under v2.)"
)

PROMPT_66_SHAPE_FINDING = (
    "Prompt 66 finding carried forward (outputs/prototypes/XT_TARGET_PROTOTYPE_V2.md section 2.3): "
    "target_xt_delta_v2's mean flips from v1's slightly negative -0.00132 to a clearly positive +0.00338 -- on "
    "average, active defensive actions now genuinely deny threat, the football-sensible direction v1 never "
    "achieved. But the shape got HARDER, not easier: skew moves from +0.093 (near-symmetric) to -0.306 (mildly "
    "left-skewed) and excess kurtosis from 8.85 to 18.33 (much heavier tails), while the exactly-zero share "
    "rises from 11.3% to 20.2% and the negative share falls from 47.0% to 35.4%. The heavier tails are a "
    "direct, understood consequence of injecting real shot_statsbomb_xg values (up to 0.897) as xt_after for "
    "the 954 rows -- 1.9% of continuing-possession rows -- where a Shot immediately follows the defensive "
    "action; the 96-cell xT grid's own maximum cell is only 0.2575, so it could never produce such values. "
    "This REINFORCES rather than resolves Prompt 64's open flag that the existing hurdle architecture does not "
    "fit this target's shape as-is: v2 is if anything less like that shape than v1 was."
)

PROMPT_66_POSSESSION_COLLAPSE_FINDING = (
    "Prompt 66 finding carried forward (outputs/prototypes/XT_TARGET_PROTOTYPE_V2.md section 2.1): on the two "
    "possession-ending slices, target_xt_delta_v2 collapses EXACTLY onto xt_before's own distribution -- "
    "action_ended_possession == True (n=4,656) mean +0.03051, action_won_possession == True (n=3,600) mean "
    "+0.03066, both confirmed identical to the same slice's xt_before mean with np.allclose. This is logically "
    "REQUIRED, not a new degenerate finding: xt_after is forced to 0.0 for exactly these rows, so the delta is "
    "xt_before by construction -- '100% of the threat that existed a moment ago is denied'. Unlike the old "
    "target_future_shot_10s, which collapsed these same rows to an identical uninformative zero, this collapses "
    "to a real, row-by-row varying distribution (mean +0.031, std ~0.050): the actual danger level that was "
    "denied. Any report that shows these slices must present the identity as construction, not as discovery."
)

PROMPT_66_CORRELATION_FINDING = (
    "Prompt 66 finding carried forward (outputs/prototypes/XT_TARGET_PROTOTYPE_V2.md section 2.4): correlation "
    "with the old target_future_xg_10s improves from v1's essentially-zero -0.016 to -0.108 -- still modest, "
    "but now an order of magnitude stronger AND correctly signed: more threat denied by a defensive action is "
    "associated with less future attacking output in the next 10 seconds. v2 and v1 are themselves moderately "
    "correlated at +0.571, which is sensible given they share the identical xt_before term and differ only in "
    "xt_after."
)

PROMPT_66_CONSTRUCTION_FINDING = (
    "Prompt 66 supersedes Prompt 64/65's construction-coupling premise. Prompt 65's Leakage Audit Part D' was "
    "built on 'xt_after is looked up directly from action_x/action_y, so every location-derived locked feature "
    "shares a definitional term with half the target'. Under v2 that is NO LONGER TRUE: xt_after is 0.0 for "
    "possession-ending actions, the next event's own location otherwise, or the next event's shot xG -- the "
    "acting row's own coordinates are never used. The v1 coupling therefore dissolves and is re-measured from "
    "scratch here rather than restated."
)

# --------------------------------------------------------------------------
# Adaptation 1, re-derived for v2 (see the module docstring for the argument)
# --------------------------------------------------------------------------

_MAD_TO_STD = 1.4826  # normal-consistency constant, so the robust scale is std-comparable

_v1_target_scale = xc.target_scale


def target_scale_v2(df) -> dict:
    """Prompt 65's `target_scale`, recomputed on v2 and extended.

    The arithmetic is deliberately unchanged -- every threshold is still the
    xg suite's own ratio translated through xg's std-fraction, computed at
    runtime from the real parquet. What is ADDED is the robustness block
    that shows the std-derived threshold no longer describes v2's bulk, and
    what is REPLACED is the prose, which asserted a negative mean and a
    symmetric shape that v2 does not have.
    """
    s = dict(_v1_target_scale(df))

    t = df[xc.TARGET].dropna()
    std = s["std"]
    mean = s["mean"]
    frac = s["xg_flat_margin_as_fraction_of_xg_std"]
    fm = s["flat_margin"]

    iqr = float(t.quantile(0.75) - t.quantile(0.25))
    mad = float((t - t.median()).abs().median())
    robust_std = _MAD_TO_STD * mad
    share_within = float((t.abs() < fm).mean() * 100)

    s["target_version"] = "v2"
    s["target_source"] = "outputs/prototypes/active_binary_xt_delta_v2.parquet (joined read-only by event_id)"
    s["supersedes"] = (
        "target_xt_delta (v1, outputs/prototypes/active_binary_xt_delta.parquet). The v1 portal built against "
        f"it is preserved unchanged at reports/analysis/{V1_PORTAL_DIRNAME}/."
    )

    s["robust_scale"] = {
        "iqr": round(iqr, 8),
        "mad": round(mad, 8),
        "robust_std_1_4826_mad": round(robust_std, 8),
        "robust_flat_margin": round(frac * robust_std, 8),
        "flat_margin_over_mad": round(fm / mad, 4) if mad else None,
        "flat_margin_over_iqr": round(fm / iqr, 4) if iqr else None,
        "pct_rows_inside_flat_margin": round(share_within, 2),
        "mean_as_fraction_of_std": round(mean / std, 4) if std else None,
    }

    # Adaptation 1 is the one place the whole suite still reaches for the OLD
    # target (it borrows xg's std-fraction to set every threshold). That makes
    # this the right place to record how the two targets actually relate --
    # measured on these same rows, not quoted from the writeup.
    both = df[[xc.TARGET, xc.XG_TARGET]].dropna()
    r_old = float(both[xc.TARGET].corr(both[xc.XG_TARGET])) if len(both) else float("nan")
    s["correlation_with_old_target"] = {
        "pair": f"{xc.TARGET} vs {xc.XG_TARGET}",
        "pearson_r": round(r_old, 4),
        "n": int(len(both)),
        "v1_pearson_r": -0.016,
        "note": (
            f"Measured on this run: r={r_old:+.4f} against the old target on these same rows, versus "
            "-0.016 for v1. Recorded HERE specifically because Adaptation 1 borrows this very target's "
            "standard-deviation fraction to set every threshold in this suite -- so wherever a report cites "
            "an xg-derived threshold, this is how the two targets actually relate. " + PROMPT_66_CORRELATION_FINDING
        ),
    }

    s["threshold_transfer_warning"] = (
        "THIS THRESHOLD DOES NOT TRANSFER CLEANLY FROM v1, and that is reported rather than forced to fit. "
        "Prompt 65 tuned its translated flat_margin against a target with excess kurtosis 8.85, an IQR of "
        "0.019119 and a MAD of 0.010277; the resulting margin (0.005486) was 0.53 x that MAD and spanned 35.4% "
        f"of rows -- a genuinely narrow 'too small to matter' band. v2's tails are far heavier (excess kurtosis "
        f"{s['excess_kurtosis']}) because Prompt 66 injects real shot xG values up to 0.897 for the 954 rows "
        "where a Shot follows the defensive action, while v2's BULK is much tighter (IQR "
        f"{iqr:.6f}, MAD {mad:.6f}). Because the translation is std-derived and std is inflated by exactly "
        f"those tails, the margin barely moves ({fm:.6f}) while the distribution it is meant to describe more "
        f"than halved: it is now {fm / mad:.2f} x the MAD, {fm / iqr:.2f} x the IQR, and spans {share_within:.1f}% "
        "of all rows. A flat band covering the majority of the data is not doing its job. The literal "
        "translated margin is KEPT as the headline number so v1-vs-v2 comparisons stay like-for-like and the xg "
        "suite's convention is still the one being followed, but the MAD-derived robust alternative "
        f"({frac * robust_std:.6f}) is published beside it on every run and every shape classification in this "
        "suite should be read with this inflation in mind."
    )

    s["scale_note"] = (
        "ADAPTATION 1, RE-EXAMINED FOR v2. Prompt 65 translated the xg suite's mean-multiple threshold "
        "convention onto the xT target via xg's own standard-deviation fraction, because v1's mean was "
        "near-zero and NEGATIVE (-0.001316), which would have flipped the sign of every threshold. v2's mean is "
        f"now POSITIVE ({mean:+.6f}), so that literal justification no longer applies -- checked, not assumed. "
        "The translation is KEPT anyway, for a reason that does still hold: at "
        f"{mean:+.6f} against a std of {std:.6f} the mean is only {abs(mean) / std * 100:.1f}% of one standard "
        "deviation, so a mean-multiple threshold would still be an arbitrary sliver, and keeping the same "
        "translation is what makes this portal numerically comparable with the v1 portal it supersedes. "
        f"Mechanically: xg's own flat_margin (0.5 x mean xg = {s['xg_flat_margin_absolute']:.6f} on these same "
        f"rows) re-expressed as a fraction of xg's own std ({s['xg_std_same_rows']:.6f}) gives {frac:.6f}, "
        f"applied to target_xt_delta_v2's std ({std:.6f}) to give {fm:.6f}. Every number is computed at runtime "
        "from the real parquet. See `threshold_transfer_warning` in this same block: the std-derived margin is "
        "materially less appropriate on v2 than it was on v1, and that is stated rather than hidden. Scale-free "
        "thresholds (|Spearman rho| >= 0.05, small-n row counts, the substitutive/additive ratio cutoffs) are "
        "carried over completely unchanged and are unaffected."
    )

    s["conditional_panel_note"] = (
        "ADAPTATION 2, unchanged in method and re-measured on v2. The xg_target suite's second panel conditions "
        "on target_future_shot_10s == 1 to strip the structural zero mass. target_xt_delta_v2 has no shot "
        f"column behind it; its structural-zero analogue is the {s['pct_zero']:.1f}% of rows with an EXACTLY "
        "zero delta. That share nearly doubled from v1's 11.3%, for an understood reason: under v2 a delta is "
        "exactly zero whenever the next same-possession event lands in the same xT grid cell as the previous "
        "one, which is far more common than v1's 'the action's own location shares a cell with the previous "
        "event'. The conditional panel throughout this suite remains 'given a non-zero delta' "
        "(target_xt_delta_v2 != 0), never 'given a shot'."
    )

    s["direction_note"] = (
        "ADAPTATION 3, unchanged in method and re-measured on v2. Sign carries the football meaning: POSITIVE "
        "target_xt_delta_v2 = xT fell across the action (threat reduced, good for the defence), NEGATIVE = xT "
        f"rose. On v2 the balance shifts markedly towards the defence -- {s['pct_positive']:.1f}% positive vs "
        f"{s['pct_negative']:.1f}% negative, where v1 was 41.7% / 47.0% -- but the target is NO LONGER close to "
        f"symmetric (skew {s['skew']}, vs v1's +0.093), so two-directional framing matters more here, not less. "
        "Rankings are by |magnitude| across BOTH directions, and every table also reports the "
        "negative/positive/zero share behind its mean."
    )

    s["log_transform_note"] = (
        "NOT an adaptation, re-checked rather than inherited: the xg_target report suite never log-transforms "
        "its target anywhere, and neither did Prompt 65's v1 xT suite. The log1p framing in this project belongs "
        "to the modelling legs, not to this report suite. v2's heavier tails make a transform more tempting "
        "than it was on v1 -- and it is still NOT applied here, because doing so would silently break "
        "comparability with both the xg portal and the v1 xT portal this one supersedes. Recorded as an open "
        "question for the modelling legs, not resolved by an EDA suite."
    )

    return s


def repoint() -> None:
    """Rebind `xt_common`'s module attributes onto v2. Idempotent.

    This is the whole re-point mechanism. Because every Prompt-65 generator
    reaches the target through `xc.<attr>` rather than opening files or
    naming columns itself, rebinding here re-points the entire suite with
    zero edits to those 21 files.
    """
    xc.TARGET = TARGET_V2
    xc.XT_PARQUET = XT_PARQUET_V2
    xc.target_scale = target_scale_v2
    # Prompt 64's finding strings are embedded verbatim in every Prompt-65
    # report. Both are now superseded -- the Clearance one asserts something
    # v2 disproves -- so they are replaced at the source rather than left to
    # propagate a false statement into a v2 report.
    xc.PROMPT_64_CLEARANCE_FINDING = PROMPT_66_CLEARANCE_FINDING
    xc.PROMPT_64_SHAPE_FINDING = PROMPT_66_SHAPE_FINDING


repoint()
