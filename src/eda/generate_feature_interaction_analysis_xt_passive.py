"""CLI entrypoint: numeric x numeric interaction analysis against
`target_xt_delta_passive` -- PASSIVE pairs only. Computes the JSON and
renders its HTML in one pass.

STEP-0 DISPOSITION: genuine passive build.
`reports/analysis/xg_target/FEATURE_INTERACTION_ANALYSIS.json` covers BOTH
legs in ONE file -- confirmed by reading it: its
`thresholds.min_marginal_delta_by_dataset` block carries an entry per leg
and its 10 pairs are the 5 active plus the 5 passive entries of the shared
`PAIRS` list. This portal's existing file of that name is Prompt 67's
ACTIVE-ONLY output (5 pairs), because Prompt 65 filtered `PAIRS` down to
its active entries. Merging a passive section into it would mean editing
active-leg report content, which this pass does not do, so the passive half
is this separate `PASSIVE_*` file.

Carried over UNCHANGED: the `PAIRS` list (imported), filtered to its 5
PASSIVE entries and deliberately NOT re-ranked or re-selected -- keeping
the same pairs is the whole reason the three targets' results are
comparable. `Q = 4`, `_bin_labeled`, `SUBSTITUTIVE_RATIO_MAX = 0.3`,
`ADDITIVE_MAGNITUDE_RATIO_MAX = 2.0` and `MIN_CELL_N_FOR_DELTA = 20` are
all ratios or row counts -- scale-free AND sign-free, so they transfer to
this leg untouched. That was re-checked rather than assumed, and it is why
this report needed only one threshold recomputed.

ADAPTED: `min_marginal_delta`, the one absolute-magnitude threshold in the
test. Recomputed from THIS leg's own standard deviations (unconditional and
non-zero-delta), not transferred from the active leg's.

Usage:
    python -m src.eda.generate_feature_interaction_analysis_xt_passive
"""

from __future__ import annotations

import json

from src.eda import xt_passive_common as pc  # re-points xt_common; MUST precede the generator imports
from src.eda import xt_common as xc
from src.eda import generate_feature_interaction_analysis_xt as i1
from src.eda import generate_feature_interaction_report_xt as ir1
from src.eda.generate_feature_interaction_analysis import PAIRS, Q

JSON_PATH = xc.OUT_DIR / "PASSIVE_FEATURE_INTERACTION_ANALYSIS.json"
HTML_PATH = xc.OUT_DIR / "PASSIVE_FEATURE_INTERACTION_ANALYSIS.html"
XG_INPUT_PATH = xc.REPO_ROOT / "reports" / "analysis" / "xg_target" / "FEATURE_INTERACTION_ANALYSIS.json"
TARGET = xc.TARGET


def main() -> None:
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt()
    scale = xc.target_scale(df)

    passive_pairs = [p for p in PAIRS if p["dataset"] == "passive"]
    min_delta = scale["flat_margin"]
    min_delta_nz = scale["flat_margin_nonzero"]

    results = []
    for spec in passive_pairs:
        r = i1.analyze_pair(df, spec, min_delta, min_delta_nz)
        r["dataset"] = "passive"  # analyze_pair stamps "active"; corrected here, not in that module
        results.append(r)
        print(f"[passive] {r['feature_a']} x {r['feature_b']}: unconditional={r['classification']}, "
              f"non-zero-delta={r['classification_given_nonzero_delta']}, "
              f"sign-flip={r['within_stratum_deltas_change_sign']}")

    order = {"interactive": 0, "substitutive": 1, "additive": 2, "inconclusive": 3}
    ordered = sorted(results, key=lambda r: order[r["classification"]])
    n_agree = sum(1 for r in results if r["classification"] == r["classification_given_nonzero_delta"])

    output = {
        "dataset": "passive",
        "leg": "passive (this portal also covers the active-binary leg -- see FEATURE_INTERACTION_ANALYSIS.json)",
        "target": TARGET,
        "target_scale": scale,
        "n_rows": int(len(df)),
        "n_unique_events": int(df["event_id"].nunique()),
        "quartile_n": Q,
        "step_0_scope_note": (
            "The xg portal's FEATURE_INTERACTION_ANALYSIS covers BOTH legs in ONE file -- confirmed by reading "
            "it: its thresholds.min_marginal_delta_by_dataset block carries an entry per leg, and its 10 pairs "
            "are the 5 active plus the 5 passive entries of generate_feature_interaction_analysis.PAIRS. That "
            "one-file convention is NOT mirrored here, and the reason is stated rather than left implicit: "
            "this portal's FEATURE_INTERACTION_ANALYSIS.json is Prompt 67's ACTIVE-ONLY output and merging a "
            "passive section into it would mean editing active-leg report content that this pass leaves "
            f"frozen. The same imported PAIRS list is filtered to its {len(passive_pairs)} PASSIVE pairs -- a "
            "scope selection, not a re-selection. The pairs are NOT re-ranked for this target: keeping the "
            "same pairs is what makes the three targets' results comparable."
        ),
        "thresholds": {
            "substitutive_ratio_max": i1.SUBSTITUTIVE_RATIO_MAX,
            "additive_magnitude_ratio_max": i1.ADDITIVE_MAGNITUDE_RATIO_MAX,
            "min_cell_n_for_delta": i1.MIN_CELL_N_FOR_DELTA,
            "min_marginal_delta": round(min_delta, 8),
            "min_marginal_delta_given_nonzero_delta": round(min_delta_nz, 8),
            "active_v2_min_marginal_delta": 0.004929,
            "note": (
                "The substitutive / additive classification cutoffs are RATIOS of one delta to another, so "
                "they are scale-free AND sign-free and are carried over completely unchanged -- re-checked "
                "for this leg rather than assumed, and the check is what makes it safe to leave them alone "
                "despite an excess kurtosis of ~35. MIN_CELL_N_FOR_DELTA (20) is a row count and is likewise "
                "untouchable by tail weight. The one absolute-magnitude threshold, min_marginal_delta, is "
                "RECOMPUTED FROM THIS LEG'S OWN STANDARD DEVIATION and not transferred from the active leg: "
                f"this leg's value is {min_delta:.6f} against active v2's 0.004929, a 1.8x difference that "
                "would have mattered, because min_marginal_delta is the cutoff below which a pair is called "
                "'inconclusive'. Using active's larger margin here would have pushed genuinely classifiable "
                "pairs into 'inconclusive' and made this leg look less structured than it is. The "
                "classification test uses |marginal delta|, so a large NEGATIVE marginal delta is as "
                "classifiable as a large positive one -- which matters more on this leg than anywhere else in "
                "the portal, since its marginal deltas run negative far more often than active's."
            ),
        },
        "candidate_selection": (
            "Same PASSIVE pairs as reports/analysis/shot_target/FEATURE_INTERACTION_ANALYSIS.json and the xg "
            "mirror -- see generate_feature_interaction_analysis.py's module docstring for the full ranking "
            "derivation and football rationale per pair. All five involve this leg's own screening, option and "
            "engagement columns, which have no active counterpart, so none of them could have been run on the "
            "active half of this portal. Computed here against target_xt_delta_passive for direct "
            "comparability, with a non-zero-delta panel in place of the xg version's shot-conditional one."
        ),
        "row_grain_note": pc.PROMPT_68_SHARED_TARGET_FINDING,
        "n_pairs_total": len(results),
        "n_pairs_where_unconditional_and_conditional_agree": n_agree,
        "n_pairs_with_sign_flipping_stratum_deltas": sum(
            1 for r in results if r["within_stratum_deltas_change_sign"]),
        "summary_table": [
            {"dataset": r["dataset"], "feature_a": r["feature_a"], "feature_b": r["feature_b"],
             "classification": r["classification"],
             "classification_given_nonzero_delta": r["classification_given_nonzero_delta"],
             "within_stratum_deltas_change_sign": r["within_stratum_deltas_change_sign"]}
            for r in ordered
        ],
        "prompt_68_cross_references": {
            "distribution_shape": pc.PROMPT_68_SHAPE_FINDING,
            "shared_target_row_grain": pc.PROMPT_68_SHARED_TARGET_FINDING,
        },
        "pairs": results,
    }

    JSON_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    xg = json.loads(XG_INPUT_PATH.read_text(encoding="utf-8")) if XG_INPUT_PATH.exists() else None
    html = ir1.build_report(output, xg)
    for needle, replacement in [
        ("FEATURE INTERACTION ANALYSIS -- XT DELTA &amp;middot; ACTIVE-BINARY LEG ONLY",
         "FEATURE INTERACTION ANALYSIS -- XT DELTA &amp;middot; PASSIVE LEG"),
        ("Numeric x Numeric Interaction Analysis (xT delta)",
         "Numeric x Numeric Interaction Analysis (xT delta, passive)"),
    ]:
        assert needle in html, (
            f"Expected {needle!r} in generate_feature_interaction_report_xt's output so it could be relabelled "
            "for the passive leg. It is not there -- that module must have changed."
        )
        html = html.replace(needle, replacement)
    HTML_PATH.write_text(html, encoding="utf-8")

    print(f"\n{n_agree}/{len(results)} pairs classify the same way unconditionally and on the "
          "non-zero-delta subset.")
    print(f"Wrote {JSON_PATH} and {HTML_PATH}")


if __name__ == "__main__":
    main()
