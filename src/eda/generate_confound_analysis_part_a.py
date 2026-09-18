"""CLI entrypoint: prompt 24 Part A -- extend reports/analysis/shot_target/CONFOUND_ANALYSIS.json
with 5 new tests, reusing generate_confound_analysis.py's exact method
(quartile the marginal, quartile the confound, stratify, check whether the
gradient survives) without altering the 2 existing tests in that file.

New tests:
  3. top_option_2_threat_score vs zone_defensive_value (passive)
  4. top_option_3_threat_score vs zone_defensive_value (passive)
  5. lane_screening_score_option_2 vs top_option_2_threat_score (passive)
  6. lane_screening_score_option_3 vs top_option_3_threat_score (passive)
  7. defenders_within_10m vs is_in_attacking_box (active) -- boolean confound
     (correlates more strongly with the marginal than is_in_defending_box,
     checked directly: 0.283 vs 0.240), so stratified by True/False rather
     than quartiled.

Tests 3/4 feed directly into the still-open Cluster 5 decision (the three
top_option_*_threat_score pairs' redundancy) -- that connection is stated
in each result's `cluster_5_connection` field, but this script does not
make the Cluster 5 call itself.

Usage:
    python -m src.eda.generate_confound_analysis_part_a
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_confound_analysis import QCUT_N, _marginal_table, _stratified_table, _verdict

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CONFOUND_ANALYSIS.json"

TARGET = "target_future_shot_10s"

TEST_3 = {
    "name": "top_option_2_threat_score_vs_zone_defensive_value",
    "title": "top_option_2_threat_score U-shape vs zone_defensive_value",
    "marginal_col": "top_option_2_threat_score",
    "marginal_labels": ["Q1 lowest", "Q2", "Q3", "Q4 highest"],
    "confound_col": "zone_defensive_value",
    "confound_labels": ["Z1 calmest", "Z2", "Z3", "Z4 most dangerous"],
    "proposed_confound_reason": "the U-shape in top_option_2_threat_score is just zone_defensive_value's own established U-shape in different clothing",
    "cluster_5_connection": (
        "Feeds directly into the still-open Cluster 5 decision (whether top_option_1/2/3_threat_score carry "
        "independent signal or are redundant with each other -- see REVIEW_METHODOLOGY.html/_V2.html and "
        "feature_config.py's PASSIVE_COLLAPSE_OPTION_RANKS). This test does not decide Cluster 5 -- it only "
        "checks whether THIS feature's pattern is explained by zone_defensive_value specifically, a separate "
        "question from whether it's redundant with top_option_1/3_threat_score."
    ),
}

TEST_4 = {
    "name": "top_option_3_threat_score_vs_zone_defensive_value",
    "title": "top_option_3_threat_score U-shape vs zone_defensive_value",
    "marginal_col": "top_option_3_threat_score",
    "marginal_labels": ["Q1 lowest", "Q2", "Q3", "Q4 highest"],
    "confound_col": "zone_defensive_value",
    "confound_labels": ["Z1 calmest", "Z2", "Z3", "Z4 most dangerous"],
    "proposed_confound_reason": "the U-shape in top_option_3_threat_score is just zone_defensive_value's own established U-shape in different clothing",
    "cluster_5_connection": (
        "Feeds directly into the still-open Cluster 5 decision (whether top_option_1/2/3_threat_score carry "
        "independent signal or are redundant with each other). This test does not decide Cluster 5 -- it only "
        "checks whether THIS feature's pattern is explained by zone_defensive_value specifically."
    ),
}

TEST_5 = {
    "name": "lane_screening_2_vs_top_option_2_threat_score",
    "title": "Lane screening (option 2) reversal",
    "marginal_col": "lane_screening_score_option_2",
    "marginal_labels": ["Q1 least", "Q2", "Q3", "Q4 most screened"],
    "confound_col": "top_option_2_threat_score",
    "confound_labels": ["T1 low", "T2", "T3", "T4 high"],
    "proposed_confound_reason": "a threatening option-2 only exists when danger is already high -- direct extension of the option-1 lane-screening test",
}

TEST_6 = {
    "name": "lane_screening_3_vs_top_option_3_threat_score",
    "title": "Lane screening (option 3) reversal",
    "marginal_col": "lane_screening_score_option_3",
    "marginal_labels": ["Q1 least", "Q2", "Q3", "Q4 most screened"],
    "confound_col": "top_option_3_threat_score",
    "confound_labels": ["T1 low", "T2", "T3", "T4 high"],
    "proposed_confound_reason": "a threatening option-3 only exists when danger is already high -- direct extension of the option-1 lane-screening test",
}


def run_test_boolean_confound(df: pd.DataFrame, spec: dict) -> dict:
    """Same method as run_test(), but the confound is boolean -- stratified
    by True/False directly instead of quartiled (a boolean has only 2
    values, qcut into 4 quantiles would fail/degenerate)."""
    marginal_bins, marginal_table = _marginal_table(df, spec["marginal_col"], spec["marginal_labels"])
    # The confound column may be stored as int64 (0/1) rather than real bool
    # -- map({True: ..., False: ...}) on an int64 Series silently produces
    # all-NaN (confirmed directly), so cast explicitly first.
    confound_bins = df[spec["confound_col"]].astype(bool).map({True: "True", False: "False"})
    strata = _stratified_table(df, marginal_bins, confound_bins, spec["marginal_labels"], spec["confound_labels"])
    verdict = _verdict(marginal_table, strata)

    extra = {k: v for k, v in spec.items() if k not in {
        "name", "title", "marginal_col", "marginal_labels", "confound_col", "confound_labels", "proposed_confound_reason",
    }}

    return {
        "name": spec["name"],
        "title": spec["title"],
        "marginal_column": spec["marginal_col"],
        "confound_column": spec["confound_col"],
        "proposed_confound_reason": spec["proposed_confound_reason"],
        "confound_type": "boolean (stratified True/False, not quartiled)",
        **extra,
        "marginal_table": marginal_table,
        "stratified_table": strata,
        "verdict": verdict,
    }


def run_test_numeric(df: pd.DataFrame, spec: dict) -> dict:
    marginal_bins, marginal_table = _marginal_table(df, spec["marginal_col"], spec["marginal_labels"])
    confound_bins = pd.qcut(df[spec["confound_col"]], QCUT_N, labels=spec["confound_labels"])
    strata = _stratified_table(df, marginal_bins, confound_bins, spec["marginal_labels"], spec["confound_labels"])
    verdict = _verdict(marginal_table, strata)

    result = {
        "name": spec["name"],
        "title": spec["title"],
        "marginal_column": spec["marginal_col"],
        "confound_column": spec["confound_col"],
        "proposed_confound_reason": spec["proposed_confound_reason"],
        "marginal_table": marginal_table,
        "stratified_table": strata,
        "verdict": verdict,
    }
    if "cluster_5_connection" in spec:
        result["cluster_5_connection"] = spec["cluster_5_connection"]
    return result


def main() -> None:
    existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    assert len(existing["tests"]) == 2, (
        f"Expected exactly 2 pre-existing tests before appending, found {len(existing['tests'])} -- "
        "aborting rather than risk duplicating or clobbering entries."
    )
    existing_names = {t["name"] for t in existing["tests"]}

    passive_df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])
    active_df = pd.read_parquet(REPO_ROOT / ACTIVE["parquet_path"])

    # Confound selection for test 7: check both candidates, use the stronger one.
    corr_defending_box = active_df["defenders_within_10m"].corr(active_df["is_in_defending_box"].astype(int))
    corr_attacking_box = active_df["defenders_within_10m"].corr(active_df["is_in_attacking_box"].astype(int))
    stronger_confound = "is_in_attacking_box" if abs(corr_attacking_box) >= abs(corr_defending_box) else "is_in_defending_box"

    test_7 = {
        "name": "defenders_within_10m_vs_box_proximity",
        "title": "defenders_within_10m vs box proximity",
        "marginal_col": "defenders_within_10m",
        "marginal_labels": ["Q1 fewest", "Q2", "Q3", "Q4 most"],
        "confound_col": stronger_confound,
        "confound_labels": ["False", "True"],
        "proposed_confound_reason": (
            "more nearby defenders correlating with more danger (not less) is opposite naive intuition -- "
            "plausibly because defenders converge precisely when the situation is already dangerous (box proximity)"
        ),
        "confound_selection_note": (
            f"Checked both candidates directly: corr(defenders_within_10m, is_in_defending_box)={corr_defending_box:.4f}, "
            f"corr(defenders_within_10m, is_in_attacking_box)={corr_attacking_box:.4f} -- used {stronger_confound} "
            "as the confound, the stronger correlation."
        ),
        "reliability_note": (
            "UNRELIABLE, not a clean finding: Part B (TOURNAMENT_STABILITY_CHECK.json/.html) found "
            "defenders_within_10m is a GENUINE TOURNAMENT-LEVEL DIFFERENCE, not arbitrary train/test noise -- "
            "U-shaped in WC2022, inverse-U in Euro2024 (same bin edges, same shape-classification method). This "
            "confound test below runs on the pooled 115-match population, which mixes two populations with "
            "opposite-direction patterns. Read the marginal/stratified numbers below as a description of the "
            "pooled data, not as evidence about a single, stable phenomenon."
        ),
    }

    new_tests = [
        run_test_numeric(passive_df, TEST_3),
        run_test_numeric(passive_df, TEST_4),
        run_test_numeric(passive_df, TEST_5),
        run_test_numeric(passive_df, TEST_6),
        run_test_boolean_confound(active_df, test_7),
    ]

    for t in new_tests:
        if t["name"] in existing_names:
            raise AssertionError(f"Test name {t['name']!r} collides with an existing entry -- aborting.")

    existing["tests"] = existing["tests"] + new_tests
    existing["part_a_extension_note"] = (
        "5 tests added by prompt 24 Part A (see each entry's 'name' for which). The original 2 tests "
        "(marking_tightness_vs_zone_defensive_value, lane_screening_vs_top_option_1_threat_score) are byte-for-byte "
        "unchanged above this note."
    )

    OUTPUT_PATH.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")

    print("Lane screening across all 3 options, side by side:")
    for name in ("lane_screening_vs_top_option_1_threat_score", "lane_screening_2_vs_top_option_2_threat_score", "lane_screening_3_vs_top_option_3_threat_score"):
        t = next(t for t in existing["tests"] if t["name"] == name)
        print(f"  {name}: verdict={t['verdict']['verdict']}")

    for t in new_tests:
        print(f"\n=== {t['title']} ===")
        print(f"Verdict: {t['verdict']['verdict']} -- {t['verdict']['verdict_meaning']}")

    print(f"\nWrote {OUTPUT_PATH} ({len(existing['tests'])} tests total)")


if __name__ == "__main__":
    main()
