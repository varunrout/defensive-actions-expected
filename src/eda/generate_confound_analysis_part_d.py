"""CLI entrypoint: prompt 29 -- close LEAKAGE_AUDIT.json's part_d_has_option_2_3
follow-up ("worth testing the same way as the marking-tightness/lane-screening
reversals... but that is separate follow-up work, not part of this audit").
has_option_2/has_option_3 show a large True-vs-False shot-rate gap (True
5.95%/5.89% vs False 15.13%/17.09% -- read directly from
reports/eda/LEAKAGE_AUDIT.json's part_d entries, not the prompt's slightly
different recalled figures) -- tested here with the exact confound method
already in CONFOUND_ANALYSIS.json, extended for a boolean marginal.

Method adaptation, stated explicitly per the prompt's own instruction:
has_option_2/has_option_3 are themselves boolean, not continuous, so the
marginal side is NOT quartiled like the existing tests -- it's split
directly on True/False. Everything downstream (stratifying by the
confound's quartile, checking whether the gap survives in every stratum) is
identical to the existing method; only _marginal_table's construction
differs. Confound side (top_option_1_threat_score, defender_x) is still
quartiled exactly as before.

Test 3's confound: the prompt names `zone_defensive_value` but suggests
checking feature_config.py first "per the correction Claude Code already
made in prompt 25". Checked: zone_defensive_value is still dropped in
PASSIVE (REDUNDANCY_DROPPED_PASSIVE, Spearman r=-1.0 vs
distance_to_defending_goal) and distance_to_defending_goal is ALSO dropped
(Cluster 4, defender-position) -- defender_x is the actually-locked,
model-facing sibling (same finding, same correction, as prompt 25's Pool A).
Used here for the same reason.

Usage:
    python -m src.eda.generate_confound_analysis_part_d
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import PASSIVE
from src.eda.generate_confound_analysis import QCUT_N, TARGET, _stratified_table, _verdict

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "CONFOUND_ANALYSIS.json"

TEST_1 = {
    "name": "has_option_2_vs_top_option_1_threat_score",
    "title": "has_option_2 shot-rate gap vs top_option_1_threat_score",
    "marginal_col": "has_option_2",
    "marginal_labels": ["False (no 2nd option)", "True (has 2nd option)"],
    "confound_col": "top_option_1_threat_score",
    "confound_labels": ["T1 low", "T2", "T3", "T4 high"],
    "proposed_confound_reason": (
        "a defender only 'has' a second ranked option worth recording when the primary threat is itself "
        "significant (more players committed forward = more danger already) -- the option-count effect might "
        "just be a proxy for how threatening the possession already is, not a defensive-shape signal in its own right"
    ),
}

TEST_2 = {
    "name": "has_option_3_vs_top_option_1_threat_score",
    "title": "has_option_3 shot-rate gap vs top_option_1_threat_score",
    "marginal_col": "has_option_3",
    "marginal_labels": ["False (no 3rd option)", "True (has 3rd option)"],
    "confound_col": "top_option_1_threat_score",
    "confound_labels": ["T1 low", "T2", "T3", "T4 high"],
    "proposed_confound_reason": "same hypothesis as has_option_2, extended to the third option",
}

TEST_3 = {
    "name": "has_option_2_vs_defender_x",
    "title": "has_option_2 shot-rate gap vs defender_x",
    "marginal_col": "has_option_2",
    "marginal_labels": ["False (no 2nd option)", "True (has 2nd option)"],
    "confound_col": "defender_x",
    "confound_labels": ["D1", "D2", "D3", "D4"],
    "proposed_confound_reason": (
        "fewer options might just mean the attack is closer to goal, not that the defending team's structure is "
        "doing anything. Confound feature is defender_x, not zone_defensive_value: zone_defensive_value is "
        "dropped in feature_config.py (REDUNDANCY_DROPPED_PASSIVE, r=-1.0 vs distance_to_defending_goal), and "
        "distance_to_defending_goal is ALSO dropped (Cluster 4, defender-position) -- defender_x is the "
        "actually-kept, model-facing sibling, same correction prompt 25 already made for this exact substitution."
    ),
}


def _marginal_table_boolean(df: pd.DataFrame, col: str, labels: list[str]) -> tuple[pd.Series, list[dict]]:
    """Boolean-marginal adaptation of _marginal_table: split directly on
    True/False rather than forcing a 2-value column into 4 quantiles."""
    bins = df[col].astype(bool).map({False: labels[0], True: labels[1]})
    grouped = df.groupby(bins, observed=True)[TARGET].agg(n="count", shots="sum", rate=lambda s: s.mean() * 100)
    table = [
        {"bin": str(idx), "n": int(row["n"]), "shots": int(row["shots"]), "rate": round(float(row["rate"]), 3)}
        for idx, row in grouped.reindex(labels).iterrows()
    ]
    return bins, table


def run_test_boolean_marginal(df: pd.DataFrame, spec: dict) -> dict:
    marginal_bins, marginal_table = _marginal_table_boolean(df, spec["marginal_col"], spec["marginal_labels"])
    confound_bins = pd.qcut(df[spec["confound_col"]], QCUT_N, labels=spec["confound_labels"])
    strata = _stratified_table(df, marginal_bins, confound_bins, spec["marginal_labels"], spec["confound_labels"])
    verdict = _verdict(marginal_table, strata)

    return {
        "name": spec["name"],
        "title": spec["title"],
        "marginal_column": spec["marginal_col"],
        "confound_column": spec["confound_col"],
        "marginal_type": "boolean (split True/False directly, not quartiled)",
        "proposed_confound_reason": spec["proposed_confound_reason"],
        "marginal_table": marginal_table,
        "stratified_table": strata,
        "verdict": verdict,
    }


def main() -> None:
    existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    assert len(existing["tests"]) == 7, (
        f"Expected exactly 7 pre-existing tests before appending (verified by reading the file, not the prompt's "
        f"stated count of 9), found {len(existing['tests'])} -- aborting rather than risk duplicating or "
        f"clobbering entries."
    )
    existing_names = {t["name"] for t in existing["tests"]}

    passive_df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])

    new_tests = [
        run_test_boolean_marginal(passive_df, TEST_1),
        run_test_boolean_marginal(passive_df, TEST_2),
        run_test_boolean_marginal(passive_df, TEST_3),
    ]

    for t in new_tests:
        if t["name"] in existing_names:
            raise AssertionError(f"Test name {t['name']!r} collides with an existing entry -- aborting.")

    existing["tests"] = existing["tests"] + new_tests
    existing["part_d_extension_note"] = (
        "3 tests added by prompt 29, closing LEAKAGE_AUDIT.json's part_d_has_option_2_3 follow-up item. The prompt "
        "that requested this stated the file had 9 pre-existing tests; reading the file directly found 7 (the "
        "original 2 plus prompt 24 Part A's 5) -- 3 tests appended on top of those 7, giving 10 total. All 7 "
        "pre-existing entries are byte-for-byte unchanged above this note."
    )

    OUTPUT_PATH.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")

    for t in new_tests:
        print(f"\n=== {t['title']} ===")
        print(f"Verdict: {t['verdict']['verdict']} -- {t['verdict']['verdict_meaning']}")

    print(f"\nWrote {OUTPUT_PATH} ({len(existing['tests'])} tests total)")


if __name__ == "__main__":
    main()
