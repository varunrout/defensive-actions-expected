"""CLI entrypoint: resolve the PASSIVE dataset's REVIEW-tier pairs from
CORRELATION_ANALYSIS_V1_HISTORICAL.json using xT-delta lift as evidence.
Computes the JSON and renders its HTML in one pass.

STEP-0 DISPOSITION: genuine passive build.
`reports/analysis/xg_target/REVIEW_ANALYSIS.json` covers BOTH legs in ONE
file -- confirmed by reading it: its `datasets` block carries an `active`
AND a `passive` entry. This portal's existing `REVIEW_ANALYSIS.json` /
`REVIEW_METHODOLOGY.html` are Prompt 67's ACTIVE-ONLY output. Merging a
passive section in would mean editing active-leg report content, which this
pass does not do, so the passive half is a separate `PASSIVE_*` pair of
files.

Exactly as in the xg and active xT versions, only Type 2 (boolean vs
boolean) actually uses target evidence. Types 1 (structural correlation
threshold), 3 (conceptual eta threshold) and 4 (name-pattern persistence
check) never reference a target at all, so their resolvers are imported and
reused unchanged -- which also means this report's Type-1/3/4 verdicts are
identical to the passive halves of the binary and xg reviews by
construction, not by coincidence.

Adaptation 1 applies to the two Type-2 thresholds, recomputed from THIS
leg's own standard deviation. Adaptation 3 applies to the opposite-sign
rule, which is if anything MORE meaningful here than on the active leg: the
passive target's positive and negative shares are almost exactly balanced
(36.4% / 37.2%), so a sign disagreement between two flags is a genuine
split rather than one flag simply sitting on the dominant side.

Usage:
    python -m src.eda.generate_review_analysis_xt_passive
"""

from __future__ import annotations

import json

from src.eda import xt_passive_common as pc  # re-points xt_common; MUST precede the generator imports
from src.eda import xt_common as xc
from src.eda import generate_review_analysis_xt as ra1
from src.eda import generate_review_report_xt as rr1

CORRELATION_PATH = xc.REPO_ROOT / "reports" / "analysis" / "shot_target" / "CORRELATION_ANALYSIS_V1_HISTORICAL.json"
JSON_PATH = xc.OUT_DIR / "PASSIVE_REVIEW_ANALYSIS.json"
HTML_PATH = xc.OUT_DIR / "PASSIVE_REVIEW_METHODOLOGY.html"
TARGET = xc.TARGET

NOTE = (
    "xT-delta-lift counterpart of reports/analysis/xg_target/REVIEW_ANALYSIS.json's PASSIVE half. Only Type 2 "
    "verdicts can differ from the binary- and xg-target versions; Types 1/3/4 are target-independent and their "
    "resolvers are imported and reused unchanged, so those verdicts match the other two targets' passive halves "
    "by construction rather than by coincidence. The two Type-2 thresholds are translated from the xg suite's "
    "0.5x/1.5x-mean-xg convention via xg's own standard-deviation fraction applied to THIS leg's std -- a "
    "multiple of target_xt_delta_passive's own near-zero NEGATIVE mean would be meaningless, exactly the "
    "situation Prompt 65's Adaptation 1 was written for and which the active leg's own v2 target no longer has "
    "(its mean is positive). The opposite-sign rule carries more weight on this target than on xg: a sign "
    "disagreement means the two flags disagree about the DIRECTION of the threat swing across the event, not "
    "merely about being above or below a base rate. "
    "ROW-GRAIN CAVEAT: every lift below is a difference of two means over defender-slot rows that share one "
    "target value per event, so the lifts are real but their precision is overstated by the row counts beside "
    "them. No threshold here is a significance threshold -- all three are effect-size cutoffs -- so the "
    "verdicts do not depend on that precision."
)


def main() -> None:
    xc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    correlation_data = json.loads(CORRELATION_PATH.read_text(encoding="utf-8"))

    assert "passive" in correlation_data["datasets"], (
        "CORRELATION_ANALYSIS_V1_HISTORICAL.json has no 'passive' dataset block -- the passive review cannot "
        "be resolved without its REVIEW-tier pairs."
    )
    # `resolve_active_xt` reads correlation_data["datasets"]["active"] and is
    # otherwise entirely dataset-agnostic (it takes its rows from
    # xc.load_active_xt, which this module's re-point layer has already
    # pointed at the passive parquet). A view with the passive block under
    # that key is what makes it reusable byte-for-byte; the SAVED JSON below
    # records the result honestly under "passive".
    view = {**correlation_data, "datasets": {"active": correlation_data["datasets"]["passive"]}}
    passive = ra1.resolve_active_xt(view)
    passive["dataset"] = "passive"
    passive["row_grain_note"] = pc.PROMPT_68_SHARED_TARGET_FINDING
    passive["threshold_recomputation"] = {
        "near_identical_threshold": passive["near_identical_threshold"],
        "distinct_signal_threshold": passive["distinct_signal_threshold"],
        "active_v2_near_identical_threshold": 0.004929,
        "active_v2_distinct_signal_threshold": 0.014787,
        "note": (
            "Both thresholds are RECOMPUTED from this leg's own standard deviation rather than transferred "
            "from the active leg. They are effect-size cutoffs on a lift, so transferring active's larger "
            "values would have made near-identical lifts look near-identical more often and pushed pairs "
            "toward 'drop one of the two' that this leg's own scale does not support. Neither is a "
            "significance threshold, so the row-grain caveat does not affect them."
        ),
    }

    output = {
        "generated_at": correlation_data["generated_at"],
        "source": str(CORRELATION_PATH.relative_to(xc.REPO_ROOT)).replace("\\", "/"),
        "leg": "passive (this portal also covers the active-binary leg -- see REVIEW_ANALYSIS.json)",
        "step_0_scope_note": (
            "The xg portal's REVIEW_ANALYSIS.json covers BOTH legs in ONE file (its `datasets` block carries an "
            "active and a passive entry) -- confirmed by reading it. That one-file convention is not mirrored "
            "here because this portal's REVIEW_ANALYSIS.json is Prompt 67's ACTIVE-ONLY output and merging a "
            "passive section into it would mean editing active-leg report content that this pass leaves frozen."
        ),
        "note": NOTE,
        "prompt_68_cross_references": {
            "shared_target_row_grain": pc.PROMPT_68_SHARED_TARGET_FINDING,
            "distribution_shape": pc.PROMPT_68_SHAPE_FINDING,
        },
        "datasets": {"passive": passive},
    }
    JSON_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    html = rr1.build_report(passive, output)
    for needle, replacement in [
        ("REVIEW METHODOLOGY -- XT DELTA &amp;middot; ACTIVE-BINARY LEG ONLY",
         "REVIEW METHODOLOGY -- XT DELTA &amp;middot; PASSIVE LEG"),
        ("Resolving the Review Tier (xT delta)", "Resolving the Review Tier (xT delta, passive)"),
        ("How every ACTIVE REVIEW-tier pair resolves using xT-delta lift as evidence -- the xT-target "
         "counterpart to the xg portal's REVIEW_METHODOLOGY.html, on the same reconstructed 51-era pool. "
         "Active-binary leg only.",
         "How every PASSIVE REVIEW-tier pair resolves using xT-delta lift as evidence -- the xT-target "
         "counterpart to the passive half of the xg portal's REVIEW_METHODOLOGY.html, on the same "
         "reconstructed 44-era pool."),
        ("Active-binary leg only; the passive half of the xg version's file is out of scope for this target.",
         "Passive leg; this portal's active half is in REVIEW_METHODOLOGY.html."),
        ("xT FELL across the action and the other marks rows where it ROSE",
         "xT FELL across the event and the other marks rows where it ROSE"),
        ("Verdict counts (active)", "Verdict counts (passive)"),
        ("needs_human_call -- ", "needs_human_call &mdash; "),
        ("pair(s), active", "pair(s), passive"),
        ("xT-delta lift (active)", "xT-delta lift (passive)"),
        ("active review pairs", "passive review pairs"),
    ]:
        assert needle in html, (
            f"Expected {needle!r} in generate_review_report_xt's output so it could be relabelled for the "
            "passive leg. It is not there -- that module must have changed."
        )
        html = html.replace(needle, replacement)
    HTML_PATH.write_text(html, encoding="utf-8")

    print(f"passive: {passive['n_review_pairs']} review pairs, "
          f"{len(passive['needs_human_call'])} needs_human_call, "
          f"{passive['n_type2_opposite_sign_pairs']} Type-2 opposite-sign pair(s)")
    print(f"  thresholds: near-identical {passive['near_identical_threshold']:.6f}, "
          f"distinct-signal {passive['distinct_signal_threshold']:.6f}")
    print(f"Wrote {JSON_PATH} and {HTML_PATH}")


if __name__ == "__main__":
    main()
