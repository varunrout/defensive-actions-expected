"""CLI entrypoint: numerical features vs `target_xt_delta_passive` --
PASSIVE LEG. Computes the atlas JSON and renders its HTML in one pass.

STEP-0 DISPOSITION: genuine passive build (inherently per-leg). The xg
portal ships `active_numerical_target_atlas.html/.json` and
`passive_numerical_target_atlas.html/.json` as a literal pair, because the
two datasets' numerical pools barely overlap: the passive pool carries this
leg's own screening and lane columns (`lane_screening_score_option_1/2/3`,
`top_option_*_threat_score`, `top_option_*_dx/dy/distance_from_ball/
angle_from_ball`, `marking_tightness`, `engagement_distance_to_carrier`,
`overload_score`) which have no active counterpart at all, while the active
pool carries local defender/attacker counts and spreads which have no
passive counterpart.

Reused by direct import, unchanged: `build_pool("passive")` (pool
construction is about which features exist, not which target they are
measured against), `analyze_feature` and `_bin_table` from Prompt 65's
active generator, `classify_shape`, `N_QUANTILE_BINS`,
`DISCRETE_CARDINALITY_THRESHOLD`, `RHO_THRESHOLD`, the canonical
match-grouped split, and the whole card/bin-bar visual system from Prompt
65's renderer.

Passive-specific and written here: the `UNRELIABLE_FEATURES` lookup (the
active generator hardcodes the `"active"` key), the adaptation section, the
findings cards and the footer -- all of which state active-leg conclusions
in the original.

Usage:
    python -m src.eda.generate_numerical_xt_target_passive
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.eda import xt_passive_common as pc  # re-points xt_common; MUST precede the generator imports
from src.eda import xt_common as xc
from src.eda import generate_numerical_xt_target_analysis as a1
from src.eda import generate_numerical_xt_target_reports as r1
from src.eda.feature_config import PASSIVE
from src.eda.generate_numerical_target_analysis import (
    DISCRETE_CARDINALITY_THRESHOLD,
    N_QUANTILE_BINS,
    RHO_THRESHOLD,
    UNRELIABLE_FEATURES,
    build_pool,
)
from src.eda.generate_numerical_target_reports import NUM_TARGET_CSS
from src.dax.models.splits import load_canonical_split
from src.eda.render import esc, finding_card, findings_grid, html_shell

OUT_DIR = xc.OUT_DIR
TARGET = xc.TARGET
JSON_PATH = OUT_DIR / "passive_numerical_target_atlas.json"
HTML_PATH = OUT_DIR / "passive_numerical_target_atlas.html"


def _adaptation_section_passive(data: dict) -> str:
    s = data["target_scale"]
    rs = s["robust_scale"]
    return f"""
<div class="xt-adapt">
<h3>What this target is, at what grain, and exactly what methodology changed because of it</h3>
<p><b>{esc(TARGET)} can be negative and is strongly LEFT-skewed.</b> On these same {s['n_defined']:,}
defender-slot rows: mean <b>{s['mean']:+.6f}</b>, std <b>{s['std']:.6f}</b>, skew <b>{s['skew']:+.3f}</b>,
excess kurtosis <b>{s['excess_kurtosis']:.2f}</b>, <b>{s['pct_negative']:.1f}%</b> negative /
<b>{s['pct_positive']:.1f}%</b> positive / <b>{s['pct_zero']:.1f}%</b> exactly zero. A POSITIVE value means xT
fell across the event (threat reduced -- good for the defence); a NEGATIVE value means xT rose. This is nothing
like <code>target_future_xg_10s</code>, which is strictly non-negative, zero-inflated and right-skewed, and none
of that target's one-sided framing survives unmodified.</p>
<p><b>One row per defender slot, one target value per event.</b> {esc(s['row_grain_note'])}
The practical consequence for the table below: every Spearman &rho; and every Pearson r is computed over
{s['n_defined']:,} rows drawn from {data['n_unique_events']:,} events, so its nominal n overstates the
independent information behind it by roughly a factor of eight. <b>The correlations themselves are not biased by
this</b> -- a repeated y value paired with genuinely varying x values still measures the real relationship --
but their standard errors are, so rank and magnitude are the readable parts here and statistical significance is
not. No p-value is used anywhere in this report to gate a conclusion.</p>
<p><b>What was adapted, and what was not:</b></p>
<ul>
<li><b>Threshold scale-setting statistic -- CHANGED, and recomputed for this leg rather than copied from the
  active one.</b> {esc(s['scale_note'])}
  Concretely on this run: flat_margin = <b>{s['flat_margin']:.6f}</b>, consistency range-trigger =
  <b>{s['range_trigger']:.6f}</b>.</li>
<li><b>And it does not transfer cleanly.</b> {esc(s['threshold_transfer_warning'])}</li>
<li><b>Conditional panel -- CHANGED.</b> {esc(s['conditional_panel_note'])}</li>
<li><b>Correlation / lift framing -- CHANGED to both directions.</b> {esc(s['direction_note'])}
  Every card states whether its binned curve crosses zero;
  <b>{data['n_features_whose_binned_curve_crosses_zero']}</b> of {data['n_features_analyzed']} features' curves
  do.</li>
<li><b>Log-transform -- NOTHING TO CHANGE.</b> {esc(s['log_transform_note'])}</li>
<li><b>Carried over completely unchanged</b> (all scale-free): the reconstructed pre-drop PASSIVE numerical pool
  (<code>build_pool("passive")</code>, unmodified), {N_QUANTILE_BINS} quantile deciles, the
  discrete-cardinality cutoff of {DISCRETE_CARDINALITY_THRESHOLD}, the |Spearman &rho;| &ge; {RHO_THRESHOLD}
  consistency-check trigger, <code>classify_shape()</code>'s U / inverse-U / monotonic / flat heuristic, and
  the canonical match-grouped train/test split.</li>
</ul>
<p style="margin-top:10px;"><b>Shape labels on this leg are indicative, not decided.</b> The std-derived
flat margin used by <code>classify_shape()</code> is {rs['flat_margin_over_mad']:.2f}x this leg's own MAD and
spans {rs['pct_rows_inside_flat_margin']:.1f}% of all rows, against 1.28x / 53.4% on the active leg's own v2
target and 0.53x / 35.4% on v1. The MAD-derived robust alternative ({rs['robust_flat_margin']:.6f}) is published
beside it in this report's <code>target_scale.robust_scale</code> block, and the count of features that classify
differently under it is reported in the findings above.</p>
</div>"""


def build_report_passive(data: dict) -> str:
    features = data["features"]
    max_abs_rho = max((abs(f["spearman_rho"]) for f in features if f["spearman_rho"] is not None), default=1.0) or 1.0
    pool = data["pool_construction"]
    s = data["target_scale"]
    rs = s["robust_scale"]

    findings = [
        finding_card(
            "pool",
            f"{pool['n_total']} features analyzed",
            f"{pool['n_locked']} locked + {pool['n_dropped']} dropped-but-included -- the same reconstructed "
            "pre-drop PASSIVE numerical candidate pool the passive xg and binary atlases use, run against the "
            "passive xT-delta target. It is NOT the active pool: this leg's screening, lane and option columns "
            "have no active counterpart, and the active leg's local counts and spreads have no passive one.",
        ),
        finding_card(
            "scale",
            f"mean {esc(TARGET)} = {s['mean']:+.6f}, std {s['std']:.6f}",
            f"strongly left-skewed ({s['skew']:+.3f}) and very heavy-tailed (excess kurtosis "
            f"{s['excess_kurtosis']:.1f}), {s['pct_negative']:.1f}% negative -- NOT a rate and NOT "
            "non-negative. Thresholds are translated from the xg suite via xg's own std-fraction "
            f"({s['xg_flat_margin_as_fraction_of_xg_std']:.6f}) applied to THIS leg's std: flat margin "
            f"{s['flat_margin']:.6f}, range trigger {s['range_trigger']:.6f}.",
        ),
        finding_card(
            "conditional",
            "given a non-zero delta",
            f"each card expands to a panel on the {s['n_nonzero']:,} rows whose delta is not exactly zero -- "
            "this target's analogue of the xg atlas's shot-conditional panel.",
        ),
        finding_card(
            "direction",
            f"{data['n_features_whose_binned_curve_crosses_zero']}/{data['n_features_analyzed']} binned curves cross zero",
            "reported explicitly because a feature that separates threat-reducing from threat-increasing events "
            "is doing something different from one that only varies in magnitude on one side.",
        ),
        finding_card(
            "row grain",
            f"{data['n_rows']:,} rows over {data['n_unique_events']:,} events",
            "the target is one value per event, repeated across that event's defender slots -- exactly as "
            "<code>target_future_shot_10s</code> and <code>target_future_xg_10s</code> already are on this "
            "leg. Correlations are unbiased by this; their standard errors are not, so read rank and magnitude, "
            "not significance.",
            flag=True,
        ),
        finding_card(
            "threshold caveat -- recomputed, not inherited",
            f"{data['n_features_classifying_differently_under_robust_margin']}/{data['n_features_analyzed']} "
            "features classify differently under the robust margin",
            f"the std-derived flat margin spans {rs['pct_rows_inside_flat_margin']:.1f}% of rows on this leg "
            f"({rs['flat_margin_over_mad']:.2f}x its MAD), against 53.4% / 1.28x on the active leg's v2 target "
            "and 35.4% / 0.53x on v1. The comparable active-leg figure was 24 of 31. Shape labels here are "
            "indicative, not decided.",
            flag=True,
        ),
    ]
    if data["n_features_flagged_inconsistent"]:
        findings.append(finding_card(
            "caution",
            f"{data['n_features_flagged_inconsistent']} feature(s) flagged train/test-inconsistent",
            "shape classification differs between train+val and test matches -- treat as noise, not a confirmed "
            "pattern, until re-examined.",
            flag=True,
        ))
    unreliable = [f["feature"] for f in features if f.get("unreliable_note")]
    if unreliable:
        findings.append(finding_card(
            "data issue", ", ".join(unreliable),
            "flagged unreliable pending a known data problem -- see the Distribution Atlas.", flag=True,
        ))

    cards_html = "".join(r1._feature_card(i + 1, f, max_abs_rho) for i, f in enumerate(features))

    body = _adaptation_section_passive(data) + findings_grid(findings) + f"""
<h2 class="section-title">Ranked by |Spearman &rho;| vs {esc(TARGET)}</h2>
<p class="section-note">Ranked by ABSOLUTE rho -- both directions. A strong negative rho (higher feature value
coincides with threat rising across the event) is exactly as much a finding as a strong positive one on this
target. Click a row to expand its binned mean-delta curve; bars diverge from zero, green right = threat reduced,
red left = threat increased. Green left border = locked, amber = dropped (still included, reason inside), red =
flagged unreliable.</p>
<div class="nt-list">{cards_html}</div>
"""

    return html_shell(
        eyebrow="NUMERICAL XT TARGET ATLAS · PASSIVE LEG",
        title=f"Passive / Off-Ball Defensive Positioning: Numerical Features vs {TARGET}",
        dek=(
            f"Every numerical feature's relationship to the passive xT-delta target, on the same reconstructed "
            f"pre-drop PASSIVE candidate pool ({pool['n_total']} features) the passive xg and binary atlases "
            "use, one row per visible defender-slot per on-ball attacking event. The target is computed once "
            "per event and joined on."
        ),
        stats=[
            (str(pool["n_total"]), "features analyzed"),
            (str(pool["n_locked"]), "locked"),
            (str(pool["n_dropped"]), "dropped, still shown"),
            (str(data["n_features_flagged_inconsistent"]), "train/test mismatch"),
            (str(data["n_features_whose_binned_curve_crosses_zero"]), "curves crossing zero"),
            (str(data["n_features_classifying_differently_under_robust_margin"]), "shape flips under robust margin"),
        ],
        body=body,
        footer=f"""
<p>Values are the <b>mean {esc(TARGET)}</b> per bin. This target is continuous, strongly LEFT-SKEWED and CAN BE
NEGATIVE (mean {s['mean']:+.6f}, skew {s['skew']:+.3f}, excess kurtosis {s['excess_kurtosis']:.1f},
{s['pct_negative']:.1f}% negative, {s['pct_zero']:.1f}% exactly zero) -- it is not a shot-rate percentage and
not the zero-inflated, right-skewed shape <code>target_future_xg_10s</code> has. <b>Shape classification</b>
uses the unmodified <code>classify_shape()</code> heuristic, but its flat margin ({s['flat_margin']:.6f}) is
translated from the xg suite's "0.5x overall mean" convention via xg's own standard-deviation fraction
({s['xg_flat_margin_as_fraction_of_xg_std']:.6f}) applied to THIS leg's own std, because a multiple of this
target's near-zero negative mean would be meaningless. <b>Train/test consistency</b> reuses the canonical
match-grouped split (outputs/models/splits/match_assignment.json), checked whenever |rho| &ge; {RHO_THRESHOLD}
or the bin range exceeds {s['range_trigger']:.6f}.</p>
<p><b>Prompt 68 cross-reference -- correlation with the old targets.</b>
{esc(pc.PROMPT_68_CORRELATION_FINDING)} Measured again on this run at the defender-slot grain:
r={s['correlation_with_old_target']['pearson_r']:+.4f} vs <code>target_future_xg_10s</code> and
r={s['correlation_with_old_target']['pearson_r_vs_shot_target']:+.4f} vs
<code>target_future_shot_10s</code>. That matters for reading the table above: the two old targets are what the
existing passive feature rankings were built against, so a feature that ranks highly here and not there is not
necessarily contradicting anything -- the targets are only loosely related.</p>
<p>Same output directory as this portal's ACTIVE half (<code>reports/analysis/xt_target/</code>), separate from
<code>reports/analysis/xg_target/</code> and <code>reports/analysis/shot_target/</code> -- different target,
different shape, different thresholds. Does not change feature_config.py's locked candidate list.</p>""",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_assignment = load_canonical_split()

    pool, pool_meta = build_pool("passive")
    df = xc.load_active_xt(columns=list({p["feature"] for p in pool}) + ["match_id"])
    scale = xc.target_scale(df)
    n_events = int(df["event_id"].nunique())

    # `analyze_feature` hardcodes UNRELIABLE_FEATURES.get("active", {}) -- the
    # one active-specific line in an otherwise dataset-agnostic function.
    # Rebound here by declared substitution rather than duplicating 60 lines,
    # and asserted so a change to that module fails loudly.
    assert "active" in UNRELIABLE_FEATURES, (
        "generate_numerical_target_analysis.UNRELIABLE_FEATURES no longer has an 'active' key; "
        "generate_numerical_xt_target_analysis.analyze_feature's lookup must have changed."
    )
    a1.UNRELIABLE_FEATURES = {"active": UNRELIABLE_FEATURES.get("passive", {})}

    print("=== passive (xT delta) ===")
    print(f"  reconstructed pool = {pool_meta['n_locked']} + {pool_meta['n_dropped']} = {pool_meta['n_total']}")
    print(f"  mean {TARGET}: {scale['mean']:+.6f}  std: {scale['std']:.6f}")
    print(f"  flat_margin={scale['flat_margin']:.6f}  range_trigger={scale['range_trigger']:.6f} "
          f"(std-fraction {scale['xg_flat_margin_as_fraction_of_xg_std']:.6f} translated from xg)")

    results = [a1.analyze_feature(df, entry, split_assignment, scale) for entry in pool]
    results.sort(key=lambda r: abs(r["spearman_rho"]) if r["spearman_rho"] is not None else -1, reverse=True)

    # The threshold recomputation Prompt 67 introduced, repeated for this leg
    # on its own numbers: how many features would classify differently if the
    # robust MAD-derived margin were used instead of the std-derived one?
    robust_fm = scale["robust_scale"]["robust_flat_margin"]
    n_flip = 0
    for r in results:
        if not r.get("bins"):
            continue
        alt = a1.classify_shape([b["bin"] for b in r["bins"]],
                                [b["mean_xt"] for b in r["bins"]], flat_margin=robust_fm)
        r["shape_under_robust_margin"] = alt
        r["shape_differs_under_robust_margin"] = bool(alt != r["shape"])
        n_flip += int(alt != r["shape"])

    n_inconsistent = sum(
        1 for r in results if r["consistency_check"].get("checked") and not r["consistency_check"].get("consistent")
    )
    n_crossing = sum(1 for r in results if r.get("binned_curve_crosses_zero"))

    output = {
        "dataset": "passive",
        "leg": "passive (this portal also covers the active-binary leg -- see active_numerical_target_atlas.json)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "target_source": scale["target_source"],
        "n_rows": int(len(df)),
        "n_unique_events": n_events,
        "target_scale": scale,
        "thresholds_carried_over_unchanged": {
            "n_quantile_bins": N_QUANTILE_BINS,
            "discrete_cardinality_threshold": DISCRETE_CARDINALITY_THRESHOLD,
            "rho_threshold": RHO_THRESHOLD,
            "classify_shape": "U / inverse-U / monotonic / flat heuristic, unmodified",
            "pool_construction": "generate_numerical_target_analysis.build_pool('passive'), unmodified",
            "split": "canonical match-grouped split (outputs/models/splits/match_assignment.json)",
        },
        "robust_margin_recomputation": {
            "std_derived_flat_margin": scale["flat_margin"],
            "mad_derived_robust_flat_margin": robust_fm,
            "n_features_classifying_differently": n_flip,
            "n_features_total": len(results),
            "active_v2_comparison": "24 of 31 on the active leg's own target_xt_delta_v2; 15 of 31 on v1",
            "note": (
                "Prompt 67 introduced this recomputation for the active leg after finding that the std-derived "
                "flat_margin no longer described target_xt_delta_v2's bulk. It is repeated here on this leg's "
                "OWN numbers rather than assumed to carry over, because Prompt 68 measured this target as "
                "heavier-tailed still. The std-derived margin is KEPT as the headline so the two halves of this "
                "portal stay comparable and the xg suite's convention is still the one being followed; the "
                "robust alternative is published beside it and every shape label should be read with this in "
                "mind. A naive reuse of the ACTIVE leg's absolute margin (0.004929) would have been worse than "
                "either: this leg's std is smaller, so its correctly-translated margin is 0.002736."
            ),
        },
        "pool_construction": pool_meta,
        "n_features_analyzed": len(results),
        "n_features_checked_for_consistency": sum(1 for r in results if r["consistency_check"].get("checked")),
        "n_features_flagged_inconsistent": n_inconsistent,
        "n_features_whose_binned_curve_crosses_zero": n_crossing,
        "n_features_classifying_differently_under_robust_margin": n_flip,
        "prompt_68_cross_references": {
            "distribution_shape": pc.PROMPT_68_SHAPE_FINDING,
            "shared_target_row_grain": pc.PROMPT_68_SHARED_TARGET_FINDING,
            "correlation_with_old_targets": pc.PROMPT_68_CORRELATION_FINDING,
        },
        "features": results,
    }

    JSON_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    html = build_report_passive(output)
    html = html.replace("</style>", NUM_TARGET_CSS + r1.XT_NUM_CSS + "</style>", 1)
    HTML_PATH.write_text(html, encoding="utf-8")

    print(f"  {n_inconsistent} train/test-inconsistent, {n_crossing}/{len(results)} curves cross zero, "
          f"{n_flip}/{len(results)} shapes flip under the robust margin")
    print(f"  Wrote {JSON_PATH} and {HTML_PATH}")


if __name__ == "__main__":
    main()
