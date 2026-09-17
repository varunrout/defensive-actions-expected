"""CLI entrypoint: test two marginal-shot-rate reversals against their
proposed confounds by conditioning (stratifying), rather than asserting
"probably a selection effect" without checking.

Test 1 (marking tightness): marginal shot rate falls monotonically from
tightest marking-quartile to loosest -- the reverse of naive intuition.
Proposed confound: zone_defensive_value ("tight marking only happens because
the zone's already dangerous"). Tested by stratifying on zone_defensive_value
quartile and checking whether the marking-tightness gradient survives within
each stratum.

Test 2 (lane screening): marginal shot rate rises monotonically from
least-screened to most-screened quartile. Proposed confound:
top_option_1_threat_score ("a threatening top option only exists when danger
is already high"). Tested the same way.

Non-causal framing throughout: "correlates with", never "causes" or
"explains away" as a claim of certainty -- only as a statistical description
of whether the association survives conditioning.

Usage:
    python -m src.eda.generate_confound_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "CONFOUND_ANALYSIS.json"

TARGET = "target_future_shot_10s"
QCUT_N = 4

TEST_1 = {
    "name": "marking_tightness_vs_zone_defensive_value",
    "title": "Marking tightness reversal",
    "marginal_col": "marking_tightness",
    "marginal_labels": ["Q1 tightest", "Q2", "Q3", "Q4 loosest"],
    "confound_col": "zone_defensive_value",
    "confound_labels": ["Z1 calmest", "Z2", "Z3", "Z4 most dangerous"],
    "proposed_confound_reason": "tight marking only happens because the zone's already dangerous",
}

TEST_2 = {
    "name": "lane_screening_vs_top_option_1_threat_score",
    "title": "Lane screening reversal",
    "marginal_col": "lane_screening_score_option_1",
    "marginal_labels": ["Q1 least", "Q2", "Q3", "Q4 most screened"],
    "confound_col": "top_option_1_threat_score",
    "confound_labels": ["T1 low", "T2", "T3", "T4 high"],
    "proposed_confound_reason": "a threatening top option only exists when danger is already high",
}


def _marginal_table(df: pd.DataFrame, col: str, labels: list[str]) -> tuple[pd.Series, list[dict]]:
    bins = pd.qcut(df[col], QCUT_N, labels=labels)
    grouped = df.groupby(bins, observed=True)[TARGET].agg(n="count", shots="sum", rate=lambda s: s.mean() * 100)
    table = [
        {"bin": str(idx), "n": int(row["n"]), "shots": int(row["shots"]), "rate": round(float(row["rate"]), 3)}
        for idx, row in grouped.iterrows()
    ]
    return bins, table


def _stratified_table(df: pd.DataFrame, marginal_bins: pd.Series, confound_bins: pd.Series, marginal_labels: list[str], confound_labels: list[str]) -> list[dict]:
    grouped = df.groupby([confound_bins, marginal_bins], observed=True)[TARGET].agg(n="count", shots="sum", rate=lambda s: s.mean() * 100)
    strata = []
    for stratum_label in confound_labels:
        row_bins = []
        monotonic_rates = []
        for bin_label in marginal_labels:
            key = (stratum_label, bin_label)
            if key in grouped.index:
                r = grouped.loc[key]
                n, shots, rate = int(r["n"]), int(r["shots"]), round(float(r["rate"]), 3)
            else:
                n, shots, rate = 0, 0, None
            row_bins.append({"bin": bin_label, "n": n, "shots": shots, "rate": rate})
            if rate is not None:
                monotonic_rates.append(rate)
        first_to_last_delta = (
            round(monotonic_rates[-1] - monotonic_rates[0], 3) if len(monotonic_rates) >= 2 else None
        )
        strata.append({
            "stratum": stratum_label,
            "bins": row_bins,
            "first_to_last_delta_pp": first_to_last_delta,
        })
    return strata


def _verdict(marginal_table: list[dict], strata: list[dict]) -> dict:
    marginal_delta = marginal_table[-1]["rate"] - marginal_table[0]["rate"]
    marginal_reversed = marginal_delta < 0  # rate falls from bin 1 to bin 4
    survives = []
    for stratum in strata:
        delta = stratum["first_to_last_delta_pp"]
        if delta is None:
            continue
        same_direction = (delta < 0) == marginal_reversed if marginal_reversed else (delta > 0) == (marginal_delta > 0)
        survives.append(same_direction)

    n_survive = sum(survives)
    n_strata = len(survives)
    if n_strata == 0:
        verdict = "inconclusive"
    elif n_survive == n_strata:
        verdict = "no"  # confound does NOT explain the reversal away -- it holds in every stratum
    elif n_survive == 0:
        verdict = "yes"  # confound explains the reversal away -- it disappears in every stratum
    else:
        verdict = "partially"

    return {
        "verdict": verdict,
        "verdict_meaning": {
            "no": "The reversal survives conditioning in every stratum -- stratifying on the proposed confound does not explain it away.",
            "yes": "The reversal disappears within every stratum -- consistent with the proposed confound explaining it away.",
            "partially": "The reversal survives in some strata but flattens or reverses in others -- a mixed result, not a clean confound.",
            "inconclusive": "Not enough populated strata to judge.",
        }[verdict],
        "n_strata_reversal_survives": n_survive,
        "n_strata_total": n_strata,
    }


def run_test(df: pd.DataFrame, spec: dict) -> dict:
    marginal_bins, marginal_table = _marginal_table(df, spec["marginal_col"], spec["marginal_labels"])
    confound_bins = pd.qcut(df[spec["confound_col"]], QCUT_N, labels=spec["confound_labels"])
    strata = _stratified_table(df, marginal_bins, confound_bins, spec["marginal_labels"], spec["confound_labels"])
    verdict = _verdict(marginal_table, strata)

    return {
        "name": spec["name"],
        "title": spec["title"],
        "marginal_column": spec["marginal_col"],
        "confound_column": spec["confound_col"],
        "proposed_confound_reason": spec["proposed_confound_reason"],
        "marginal_table": marginal_table,
        "stratified_table": strata,
        "verdict": verdict,
    }


def main() -> None:
    df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])

    result_1 = run_test(df, TEST_1)
    result_2 = run_test(df, TEST_2)

    output = {
        "dataset": "passive",
        "parquet_path": PASSIVE["parquet_path"],
        "n_rows": len(df),
        "target_col": TARGET,
        "methodology": (
            "Each candidate feature is cut into quartiles (pd.qcut, 4 bins). The marginal table shows shot rate "
            "per quartile, unconditioned. The stratified table repeats this within each quartile of the proposed "
            "confound. If the marginal gradient survives (same direction, comparable magnitude) inside every "
            "confound stratum, conditioning on the confound does not explain the reversal away -- it is not a pure "
            "selection artefact of that specific confound. This tests one candidate confound at a time; it does "
            "not rule out others."
        ),
        "tests": [result_1, result_2],
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n=== {result_1['title']} ===")
    print("Marginal:", [(b["bin"], b["rate"]) for b in result_1["marginal_table"]])
    print(f"Verdict: {result_1['verdict']['verdict']} -- {result_1['verdict']['verdict_meaning']}")

    print(f"\n=== {result_2['title']} ===")
    print("Marginal:", [(b["bin"], b["rate"]) for b in result_2["marginal_table"]])
    print(f"Verdict: {result_2['verdict']['verdict']} -- {result_2['verdict']['verdict_meaning']}")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
