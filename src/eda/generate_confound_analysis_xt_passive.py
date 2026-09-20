"""CLI entrypoint: confound (reversal) testing against
`target_xt_delta_passive` -- PASSIVE LEG. Computes the JSON and renders its
HTML in one pass.

STEP-0 DISPOSITION: genuine passive build, and -- like the Leakage Audit --
one where the leg asymmetry runs the OPPOSITE way from the rest of this
portal. Confirmed by reading the source, not assumed:
`reports/analysis/xg_target/CONFOUND_ANALYSIS.json` carries
`"dataset": "passive"` and holds 5 tests, ALL of them passive-dataset tests
on passive-only columns, produced by `generate_confound_analysis_xg.py`
(2 tests) and `generate_confound_analysis_part_d_xg.py` (3 tests). Prompt 65
searched the whole repo and found exactly ONE active-dataset confound test
anywhere, for any target, which is why the active half of this portal
mirrors that single test and invents two of its own.

**This half needs no invention at all.** All five established passive tests
mirror directly, same columns, same marginal/confound pairs, same labels:

  1. marking_tightness vs zone_defensive_value            (xg Test 1)
  2. lane_screening_score_option_1 vs top_option_1_threat_score (xg Test 2)
  3. has_option_2 vs top_option_1_threat_score            (xg Part D Test 1)
  4. has_option_3 vs top_option_1_threat_score            (xg Part D Test 2)
  5. has_option_2 vs defender_x                           (xg Part D Test 3)

METHOD carried over unchanged: `_verdict()` is imported from
`generate_confound_analysis` and reused BYTE-FOR-BYTE via
`generate_confound_analysis_xt.run_test_xt`. It compares only the SIGN of
first-to-last deltas across strata, never a magnitude threshold, so it is
target-scale-agnostic. That property is exactly why this report needed no
threshold recomputation despite this leg's excess kurtosis of ~35: a sign
comparison cannot be distorted by tail weight. Re-checked, not assumed.
`QCUT_N = 4` unchanged.

ADAPTED: each test carries a `given_nonzero_delta` sub-object in place of
the xg version's `given_shot` (Prompt 65's Adaptation 2), and the marginal
and stratified tables carry this target's negative/zero/positive shares
(Adaptation 3).

Usage:
    python -m src.eda.generate_confound_analysis_xt_passive
"""

from __future__ import annotations

import json

from src.eda import xt_passive_common as pc  # re-points xt_common; MUST precede the generator imports
from src.eda import xt_common as xc
from src.eda import generate_confound_analysis_xt as c1
from src.eda import generate_confound_report_xt as cr1
from src.eda.feature_config import PASSIVE
from src.eda.generate_confound_analysis import TEST_1 as XG_TEST_1, TEST_2 as XG_TEST_2
from src.eda.generate_confound_analysis_part_d import (
    TEST_1 as PD_TEST_1,
    TEST_2 as PD_TEST_2,
    TEST_3 as PD_TEST_3,
)

JSON_PATH = xc.OUT_DIR / "PASSIVE_CONFOUND_ANALYSIS.json"
HTML_PATH = xc.OUT_DIR / "PASSIVE_CONFOUND_ANALYSIS.html"
XG_INPUT_PATH = xc.REPO_ROOT / "reports" / "analysis" / "xg_target" / "CONFOUND_ANALYSIS.json"
TARGET = xc.TARGET

_MIRROR_NOTE = (
    "LITERAL MIRROR -- same marginal column, same confound column, same quartile/boolean labels, same "
    "_verdict() function, run against target_xt_delta_passive instead of the xg target. Nothing about this "
    "test was re-selected or re-ranked for this target: keeping the established specification is what makes "
    "the passive xg, binary and xT verdicts comparable cell by cell. Source: "
)

_ROW_GRAIN_NOTE = (
    "ROW-GRAIN NOTE, applying to every n in this test. The target is one value per event_id repeated across "
    "that event's visible defender slots, so a stratum's n is a row count, not a count of independent "
    "observations. The confound and marginal columns DO vary within an event (they are per-defender-slot "
    "measurements), so the stratification itself is meaningful -- what is inflated is only the apparent "
    "precision of each cell's mean. No verdict here uses a p-value; _verdict() compares signs only."
)


def _spec(base: dict, source: str, marginal_type: str) -> dict:
    """Take an established xg/binary passive test spec and attach the fields
    `run_test_xt` needs, without altering any of its column or label
    choices."""
    return {
        **base,
        "marginal_type": marginal_type,
        "confound_type": "numeric",
        "provenance": _MIRROR_NOTE + source,
        "is_new_test": False,
        "reliability_note": _ROW_GRAIN_NOTE,
        "prompt_64_link": pc.PROMPT_68_SHARED_TARGET_FINDING,
    }


def main() -> None:
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt()

    specs = [
        _spec(XG_TEST_1, "generate_confound_analysis.TEST_1, mirrored into the xg portal by "
                         "generate_confound_analysis_xg.py", "numeric"),
        _spec(XG_TEST_2, "generate_confound_analysis.TEST_2, mirrored into the xg portal by "
                         "generate_confound_analysis_xg.py", "numeric"),
        _spec(PD_TEST_1, "generate_confound_analysis_part_d.TEST_1, mirrored into the xg portal by "
                         "generate_confound_analysis_part_d_xg.py", "boolean"),
        _spec(PD_TEST_2, "generate_confound_analysis_part_d.TEST_2, mirrored into the xg portal by "
                         "generate_confound_analysis_part_d_xg.py", "boolean"),
        _spec(PD_TEST_3, "generate_confound_analysis_part_d.TEST_3, mirrored into the xg portal by "
                         "generate_confound_analysis_part_d_xg.py", "boolean"),
    ]

    tests = [c1.run_test_xt(df, s) for s in specs]

    output = {
        "dataset": "passive",
        "leg": "passive (this portal also covers the active-binary leg -- see CONFOUND_ANALYSIS.json)",
        "parquet_path": PASSIVE["parquet_path"],
        "n_rows": len(df),
        "n_unique_events": int(df["event_id"].nunique()),
        "target_col": TARGET,
        "target_scale": xc.target_scale(df),
        "step_0_scope_note": (
            "CONFIRMED BY READING THE SOURCE, and it points the OPPOSITE way from most scope notes in this "
            "portal: reports/analysis/xg_target/CONFOUND_ANALYSIS.json carries \"dataset\": \"passive\" and all "
            "5 of its tests are passive-dataset tests on passive-only columns (marking_tightness, "
            "zone_defensive_value, lane_screening_score_option_1, top_option_1_threat_score, has_option_2/3, "
            "defender_x). Confound testing in this project is an established PASSIVE methodology; it is the "
            "ACTIVE leg that has exactly one test anywhere in the repo, which is why the active half of this "
            "portal had to invent two of its own. All five established tests mirror here with no substitution "
            "and no invention -- this report contains no new test."
        ),
        "methodology": (
            "Same quartile/boolean-stratification method as the binary and xg confound analyses. _verdict() is "
            "imported from generate_confound_analysis and reused BYTE-FOR-BYTE through "
            "generate_confound_analysis_xt.run_test_xt -- it compares only the SIGN of first-to-last deltas "
            "across strata, never a magnitude threshold, so it is target-scale-agnostic. That property was "
            "RE-CHECKED rather than assumed for this leg and is exactly why this report needed no threshold "
            "recomputation despite an excess kurtosis of ~35, the heaviest tail anywhere in this portal: a "
            "sign comparison cannot be distorted by tail weight. QCUT_N=4 unchanged. Only the table builders "
            "differ from the xg version: mean target_xt_delta_passive instead of mean xG, with the "
            "negative/zero/positive shares added (Adaptation 3). Each test carries a 'given_nonzero_delta' "
            "sub-object in place of the xg version's 'given_shot' -- this target's structural-zero analogue "
            "(Adaptation 2), and on this leg that zero mass is 26.4% of rows, larger than active's 20.2%."
        ),
        "threshold_note": (
            "NO THRESHOLD IN THIS REPORT NEEDED PASSIVE-SPECIFIC RECOMPUTATION, and that is a checked result "
            "rather than an omission. The only tunable quantities here are QCUT_N (a count) and _verdict()'s "
            "sign comparison (sign-free of magnitude). Neither can be distorted by this leg's heavier tails. "
            "The reports in this portal that DID need the flat_margin recomputed are the Numerical Target "
            "Atlas, the Tournament Stability Check, both Slice Stratifications and the Feature Interaction "
            "Analysis -- every one of which uses classify_shape() or an absolute marginal-delta cutoff."
        ),
        "row_grain_note": pc.PROMPT_68_SHARED_TARGET_FINDING,
        "prompt_68_cross_references": {
            "shared_target_row_grain": pc.PROMPT_68_SHARED_TARGET_FINDING,
            "distribution_shape": pc.PROMPT_68_SHAPE_FINDING,
            "correlation_with_old_targets": pc.PROMPT_68_CORRELATION_FINDING,
            "no_v1_detour": pc.PROMPT_68_NO_V1_DETOUR_FINDING,
        },
        "tests": tests,
    }

    JSON_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    # Render with the active xT renderer, which is entirely data-driven apart
    # from four label literals naming the active leg / Prompt 64. Those are
    # corrected by declared, asserted substitution rather than duplicating a
    # 210-line renderer -- the same trade generate_feature_lock_report_xt_v2.py
    # makes. The xg comparison is passed as the "binary" reference so each
    # test shows how the same cell resolved against target_future_xg_10s.
    xg = json.loads(XG_INPUT_PATH.read_text(encoding="utf-8")) if XG_INPUT_PATH.exists() else None
    html = cr1.build_report(output, xg)
    for needle, replacement in [
        # render_article escapes the eyebrow, so the literal in the rendered
        # HTML is the &amp;-escaped form -- matched here as it actually appears.
        ("CONFOUND ANALYSIS -- XT DELTA &amp;middot; ACTIVE-BINARY LEG ONLY",
         "CONFOUND ANALYSIS -- XT DELTA &amp;middot; PASSIVE LEG"),
        ("Active Dataset Confound (Reversal) Testing (xT delta)",
         "Passive Dataset Confound (Reversal) Testing (xT delta)"),
        ("carried forward from Prompt 64", "carried forward from Prompt 68"),
        ("Clearance / action_x artefact", "one target value per event, shared across defender slots"),
        ("reliability caveat", "row-grain caveat"),
        ("read the numbers as pooled-population description", "n is a row count, not an independent-sample count"),
        ("rows (active)", "rows (passive)"),
        ("diverge from binary-target verdict", "diverge from the passive xG verdict"),
        ("Binary target (<code>target_future_shot_10s</code>) verdict",
         "Passive xG target (<code>target_future_xg_10s</code>) verdict"),
        ("agrees with the binary-target version of this same test",
         "agrees with the passive xG version of this same test"),
        ("diverges from the binary-target version of this same test",
         "diverges from the passive xG version of this same test"),
        ("Both the binary target and xT delta return", "Both the passive xG target and xT delta return"),
        ("One test mirrored from the only active confound test that exists in this repo, one new test "
         "motivated by Prompt 64's Clearance/action_x finding.",
         "All five established passive confound tests mirrored literally -- same columns, same labels, no "
         "invention and no substitution, because confound testing in this project is a passive methodology."),
        ("1", "5") if False else ("mirrored / 1 new", "mirrored / 0 new"),
    ]:
        assert needle in html, (
            f"Expected {needle!r} in generate_confound_report_xt's output so it could be relabelled for the "
            "passive leg. It is not there -- that module must have changed. Refusing to publish a passive "
            "report carrying active-leg labels."
        )
        html = html.replace(needle, replacement)
    HTML_PATH.write_text(html, encoding="utf-8")

    for t in tests:
        print(f"\n=== {t['title']} (xT delta, passive) ===")
        print("Marginal:", [(b["bin"], b["rate"]) for b in t["marginal_table"]])
        print(f"Verdict: {t['verdict']['verdict']} "
              f"({t['verdict']['n_strata_reversal_survives']}/{t['verdict']['n_strata_total']} strata)")
        g = t["given_nonzero_delta"]
        print(f"Given non-zero delta (n={g['n_rows_used']:,}): {g['verdict']['verdict']} "
              f"({g['verdict']['n_strata_reversal_survives']}/{g['verdict']['n_strata_total']})")

    print(f"\nWrote {JSON_PATH} and {HTML_PATH}")


if __name__ == "__main__":
    main()
