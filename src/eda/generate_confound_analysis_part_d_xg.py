"""CLI entrypoint: xG counterpart to generate_confound_analysis_part_d.py --
same 3 has_option_2/has_option_3 confound tests, against mean xG instead of
shot-rate percentage, each with a given_shot sub-object (same pattern
generate_confound_analysis_xg.py already established).

reports/analysis/xg_target/CONFOUND_ANALYSIS.json currently only has the original 2
tests mirrored (never received prompt 24 Part A's 5 additions) -- this is
the first Part-A-style extension applied to that file, noted explicitly in
the output rather than left to look like an established pattern.

Usage:
    python -m src.eda.generate_confound_analysis_part_d_xg
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import PASSIVE
from src.eda.generate_confound_analysis import QCUT_N, _verdict
from src.eda.generate_confound_analysis_part_d import TEST_1, TEST_2, TEST_3
from src.eda.generate_confound_analysis_xg import SHOT_COL, TARGET_XG, _stratified_table_xg

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "xg_target" / "CONFOUND_ANALYSIS.json"


def _marginal_table_xg_boolean(df: pd.DataFrame, col: str, labels: list[str]) -> tuple[pd.Series, list[dict]]:
    bins = df[col].astype(bool).map({False: labels[0], True: labels[1]})
    grouped = df.groupby(bins, observed=True)[TARGET_XG].agg(n="count", xg_sum="sum", mean_xg="mean")
    table = [
        {"bin": str(idx), "n": int(row["n"]), "xg_sum": round(float(row["xg_sum"]), 3), "rate": round(float(row["mean_xg"]), 6)}
        for idx, row in grouped.reindex(labels).iterrows()
    ]
    return bins, table


def run_test_xg_boolean_marginal(df: pd.DataFrame, spec: dict, given_shot: bool = False) -> dict:
    marginal_bins, marginal_table = _marginal_table_xg_boolean(df, spec["marginal_col"], spec["marginal_labels"])
    confound_bins = pd.qcut(df[spec["confound_col"]], QCUT_N, labels=spec["confound_labels"])
    strata = _stratified_table_xg(df, marginal_bins, confound_bins, spec["marginal_labels"], spec["confound_labels"])
    verdict = _verdict(marginal_table, strata)

    result = {
        "name": spec["name"],
        "title": spec["title"],
        "marginal_column": spec["marginal_col"],
        "confound_column": spec["confound_col"],
        "marginal_type": "boolean (split True/False directly, not quartiled)",
        "proposed_confound_reason": spec["proposed_confound_reason"],
        "n_rows_used": len(df),
        "marginal_table": marginal_table,
        "stratified_table": strata,
        "verdict": verdict,
    }
    if not given_shot:
        shot_df = df.loc[df[SHOT_COL] == 1]
        result["given_shot"] = run_test_xg_boolean_marginal(shot_df, spec, given_shot=True)
    return result


def main() -> None:
    existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    assert len(existing["tests"]) == 2, (
        f"Expected exactly 2 pre-existing tests before appending (this file never received prompt 24 Part A's "
        f"5 additions, verified by reading it directly), found {len(existing['tests'])} -- aborting."
    )
    existing_names = {t["name"] for t in existing["tests"]}

    df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])

    new_tests = [
        run_test_xg_boolean_marginal(df, TEST_1),
        run_test_xg_boolean_marginal(df, TEST_2),
        run_test_xg_boolean_marginal(df, TEST_3),
    ]
    for t in new_tests:
        if t["name"] in existing_names:
            raise AssertionError(f"Test name {t['name']!r} collides with an existing entry -- aborting.")

    existing["tests"] = existing["tests"] + new_tests
    existing["part_d_extension_note"] = (
        "3 tests added by prompt 29 (xG mirror of the binary-target has_option_2/3 confound tests). "
        "reports/analysis/xg_target/CONFOUND_ANALYSIS.json had never received prompt 24 Part A's 5-test extension -- this is "
        "the FIRST Part-A-style extension applied to this file, not a continuation of an established xg pattern. "
        "The 2 pre-existing entries above this note are byte-for-byte unchanged."
    )

    OUTPUT_PATH.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")

    for t in new_tests:
        print(f"\n=== {t['title']} (xG) ===")
        print(f"Verdict: {t['verdict']['verdict']} -- {t['verdict']['verdict_meaning']}")
        gs = t["given_shot"]
        print(f"Given-shot verdict (n={gs['n_rows_used']:,}): {gs['verdict']['verdict']} -- {gs['verdict']['verdict_meaning']}")

    print(f"\nWrote {OUTPUT_PATH} ({len(existing['tests'])} tests total)")


if __name__ == "__main__":
    main()
