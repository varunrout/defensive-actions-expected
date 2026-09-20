"""CLI entrypoint: confound (reversal) testing against `target_xt_delta_v2`,
ACTIVE dataset.

STEP-0 DISPOSITION: REWRITE.

All of the MACHINERY is reused by direct import from
`generate_confound_analysis_xt` -- `run_test_xt`, `_agg`,
`_marginal_table_xt`, `_stratified_table_xt`, `_bin_series`,
`_confound_series` -- which in turn reuses `_verdict()` from
`generate_confound_analysis` byte-for-byte and `QCUT_N = 4` unchanged.
`_verdict()` compares only the SIGN of first-to-last deltas across strata,
never a magnitude threshold, so it is target-scale-agnostic and transfers to
v2 untouched. That is the same reasoning Prompt 65 used, re-checked rather
than assumed: it survives v2's heavier tails precisely because it never
looks at a magnitude.

What needed rewriting is the TEST SET, because Test 2's premise is gone:

  Test 1 -- UNCHANGED in specification. Still the exact mirror of the only
    active-dataset confound test that exists anywhere in this repo
    (`defenders_within_10m_vs_box_proximity`, from
    generate_confound_analysis_part_a.py). Same marginal column, same
    quartile labels, same live confound-selection rule, same reliability
    caveat. Re-pointed at v2 and nothing else.

  Test 2 -- RE-FRAMED, same columns. Prompt 65's Test 2 asked whether
    "Clearance's NEGATIVE mean delta survives conditioning on defending-box
    proximity". Under v2 Clearance's mean delta is POSITIVE (+0.0247, 1st
    of 8), so that question is no longer about anything that exists. The
    same two columns are kept -- so the v1 and v2 portals remain directly
    comparable on the same cells -- but the question is restated to what
    the data now poses: is Clearance's newly POSITIVE mean a real
    event-type effect, or is it a location effect in the opposite
    direction?

  Test 3 -- NEW, and labelled new. It exists because Prompt 66's correction
    introduced a mechanism that did not exist under v1: possession-ending
    actions get `xt_after = 0` by construction, so their delta is
    `xt_before` and is non-negative wherever xt_before is. Clearances end
    their possession far more often than average, so Clearance's positive
    mean could be entirely a construction effect rather than anything about
    clearing. Conditioning on `action_ended_possession` is the established
    instrument in this suite for exactly that question, so it is applied
    rather than a new one invented.

Usage:
    python -m src.eda.generate_confound_analysis_xt_v2
"""

from __future__ import annotations

import json

from src.eda import xt_v2_common as v2  # re-points xt_common; MUST precede the generator import
from src.eda import xt_common as xc
from src.eda import generate_confound_analysis_xt as c1

OUTPUT_PATH = xc.OUT_DIR / "CONFOUND_ANALYSIS.json"
TARGET = xc.TARGET


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt()

    # --- Test 1: unchanged mirror, confound selection recomputed live ---
    corr_def = float(df["defenders_within_10m"].corr(df["is_in_defending_box"].astype(int)))
    corr_att = float(df["defenders_within_10m"].corr(df["is_in_attacking_box"].astype(int)))
    stronger = "is_in_attacking_box" if abs(corr_att) >= abs(corr_def) else "is_in_defending_box"

    test_1 = {
        "name": "defenders_within_10m_vs_box_proximity",
        "title": "defenders_within_10m vs box proximity",
        "marginal_col": "defenders_within_10m",
        "marginal_type": "numeric",
        "marginal_labels": ["Q1 fewest", "Q2", "Q3", "Q4 most"],
        "confound_col": stronger,
        "confound_type": "boolean",
        "confound_labels": ["False", "True"],
        "proposed_confound_reason": (
            "more nearby defenders correlating with more danger (not less) is opposite naive intuition -- "
            "plausibly because defenders converge precisely when the situation is already dangerous (box proximity)"
        ),
        "provenance": (
            "MIRROR of the only active-dataset confound test that exists anywhere in this repo: "
            "reports/analysis/shot_target/CONFOUND_ANALYSIS.json's 'defenders_within_10m_vs_box_proximity', "
            "added by generate_confound_analysis_part_a.py. The xg portal never received it. Same marginal "
            "column, same quartile labels, same live confound-selection rule, same reliability caveat. This "
            "test's SPECIFICATION is unchanged from the v1 edition of this report -- only the target it runs "
            "against differs, so the two portals' cells are directly comparable."
        ),
        "confound_selection_note": (
            f"Checked both candidates directly on this run: corr(defenders_within_10m, is_in_defending_box)="
            f"{corr_def:.4f}, corr(defenders_within_10m, is_in_attacking_box)={corr_att:.4f} -- used "
            f"{stronger}, the stronger correlation. Same rule the binary-target version applies."
        ),
        "reliability_note": (
            "UNRELIABLE, not a clean finding -- caveat carried forward from the binary-target version and "
            "re-checked against this portal's own Tournament Stability Check: defenders_within_10m is a "
            "GENUINE TOURNAMENT-LEVEL DIFFERENCE, not arbitrary train/test noise. This test runs on the pooled "
            "115-match population, which mixes two populations. Read the numbers below as a description of the "
            "pooled data, not as evidence about a single stable phenomenon."
        ),
    }

    # --- Test 2: same cells as v1, question re-framed because the sign flipped ---
    test_2 = {
        "name": "clearance_vs_defending_box_proximity",
        "title": "Clearance's POSITIVE mean delta vs defending-box proximity",
        "marginal_col": "event_type",
        "marginal_type": "category_indicator",
        "marginal_category": xc.CLEARANCE_EVENT_TYPE,
        "marginal_labels": ["not Clearance", "Clearance"],
        "confound_col": "is_in_defending_box",
        "confound_type": "boolean",
        "confound_labels": ["False", "True"],
        "proposed_confound_reason": (
            "clearances happen overwhelmingly deep in the defending box. Under v1 that location was scored AS "
            "xt_after, which is what made Clearance look like the worst event type. Under v2 the location is no "
            "longer scored at all, but it still selects WHICH situations become clearances -- so the question "
            "is whether Clearance's now-positive mean survives conditioning on where it happens, or whether it "
            "is a defending-box effect wearing a Clearance label"
        ),
        "provenance": (
            "RE-FRAMED, not re-pointed. The v1 edition of this report asked whether Clearance's NEGATIVE mean "
            "delta survives conditioning on defending-box proximity, and reached a 'partially' verdict. Under "
            "the corrected xt_after, Clearance's mean delta is POSITIVE and it ranks 1st of 8 rather than 8th, "
            "so the v1 question is no longer about anything that exists in the data. The same two columns and "
            "the same method are kept deliberately, so the v1 and v2 portals can be compared cell by cell; only "
            "the question the cells are asked has been restated to match the direction the data now runs in."
        ),
        "is_new_test": False,
        "prompt_64_link": v2.PROMPT_66_CLEARANCE_FINDING,
    }

    # --- Test 3: NEW for v2 -- the construction confound v1 could not have ---
    test_3 = {
        "name": "clearance_vs_possession_ending_construction",
        "title": "Clearance's positive mean delta vs the xt_after = 0 construction rule",
        "marginal_col": "event_type",
        "marginal_type": "category_indicator",
        "marginal_category": xc.CLEARANCE_EVENT_TYPE,
        "marginal_labels": ["not Clearance", "Clearance"],
        "confound_col": "action_ended_possession",
        "confound_type": "boolean",
        "confound_labels": ["False", "True"],
        "proposed_confound_reason": (
            "Prompt 66 sets xt_after = 0 for every action that ends its own possession, so those rows' delta is "
            "xt_before by construction and is positive wherever xt_before is. Clearances end their possession "
            "far more often than the average defensive action. Clearance's newly positive mean could therefore "
            "be a CONSTRUCTION effect -- an artefact of the fix -- rather than a football finding. This test "
            "asks whether the positive mean survives INSIDE each possession-outcome stratum, where the "
            "construction rule is held constant"
        ),
        "provenance": (
            "NEW TEST -- it has no counterpart in the v1 xT portal, the xg portal or the binary portal, and is "
            "labelled as new rather than presented as established. It could not have existed under v1, because "
            "the mechanism it tests (xt_after forced to 0 by a possession flag) was introduced by Prompt 66. "
            "The confound-testing method itself is the established instrument in this suite for asking whether "
            "a marginal pattern survives conditioning, so it is applied unchanged rather than replaced."
        ),
        "is_new_test": True,
        "reliability_note": (
            "Read the True stratum with care: inside action_ended_possession == True every row's delta EQUALS "
            "its own xt_before by construction (verified in this portal's Leakage Audit Part C' with "
            "np.allclose). That stratum is therefore not an independent measurement of clearing -- it is a "
            "measurement of how much threat existed a moment earlier. The informative comparison is the False "
            "stratum, where the construction rule does not apply."
        ),
        "prompt_64_link": v2.PROMPT_66_POSSESSION_COLLAPSE_FINDING,
    }

    tests = [c1.run_test_xt(df, test_1), c1.run_test_xt(df, test_2), c1.run_test_xt(df, test_3)]

    output = {
        "dataset": "active",
        "leg": "active-binary only (passive leg out of scope for this target)",
        "parquet_path": "data/features/player_defensive_actions.parquet",
        "n_rows": len(df),
        "target_col": TARGET,
        "target_scale": xc.target_scale(df),
        "step_0_scope_note": (
            "CONFIRMED BY READING THE SOURCE, unchanged from the v1 edition: reports/analysis/xg_target/"
            "CONFOUND_ANALYSIS.json's 5 tests are ALL passive-dataset tests on passive-only columns. Exactly one "
            "active-dataset confound test exists anywhere in this repo for any target -- "
            "'defenders_within_10m_vs_box_proximity' in the binary portal, from "
            "generate_confound_analysis_part_a.py; the xg portal never received it. Test 1 mirrors that test "
            "exactly."
        ),
        "v2_rebuild_note": (
            "Test 1's specification is unchanged from the v1 edition. Test 2 keeps v1's columns and method but "
            "RESTATES ITS QUESTION: v1 asked whether Clearance's negative mean delta survives conditioning, and "
            "under the corrected xt_after that mean is positive, so the v1 question is about nothing. Test 3 is "
            "NEW and tests a mechanism Prompt 66 introduced -- whether Clearance's positive mean is itself an "
            "artefact of the xt_after = 0 construction rule. The v1 edition is preserved at "
            f"reports/analysis/{v2.V1_PORTAL_DIRNAME}/CONFOUND_ANALYSIS.html."
        ),
        "methodology": (
            "Same quartile/boolean-stratification method as the binary and xg confound analyses. _verdict() is "
            "imported from generate_confound_analysis and reused BYTE-FOR-BYTE -- it compares only the SIGN of "
            "first-to-last deltas across strata, never a magnitude threshold, so it is target-scale-agnostic. "
            "That property was re-checked rather than assumed for v2 and is exactly why this report needed no "
            "threshold adjustment despite v2's much heavier tails: a sign comparison cannot be distorted by "
            "kurtosis. QCUT_N=4 unchanged. Table builders and the 'given_nonzero_delta' sub-object are imported "
            "unchanged from the v1 edition's generate_confound_analysis_xt."
        ),
        "prompt_66_cross_references": {
            "clearance_reversal": v2.PROMPT_66_CLEARANCE_FINDING,
            "distribution_shape": v2.PROMPT_66_SHAPE_FINDING,
            "possession_ending_collapse": v2.PROMPT_66_POSSESSION_COLLAPSE_FINDING,
        },
        "tests": tests,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    for t in tests:
        print(f"\n=== {t['title']} (xT delta v2) ===")
        print("Marginal:", [(b["bin"], b["rate"]) for b in t["marginal_table"]])
        print(f"Verdict: {t['verdict']['verdict']} "
              f"({t['verdict']['n_strata_reversal_survives']}/{t['verdict']['n_strata_total']} strata)")
        g = t["given_nonzero_delta"]
        print(f"Given non-zero delta (n={g['n_rows_used']:,}): {g['verdict']['verdict']} "
              f"({g['verdict']['n_strata_reversal_survives']}/{g['verdict']['n_strata_total']})")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
