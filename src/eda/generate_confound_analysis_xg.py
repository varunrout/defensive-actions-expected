"""CLI entrypoint: xG counterpart to generate_confound_analysis.py -- same
two reversal tests (marking-tightness vs zone_defensive_value,
lane-screening vs top_option_1_threat_score), same quartile-stratification
method, against mean xG instead of shot-rate percentage.

_verdict() is reused unchanged from the binary-target script: it only
compares the SIGN of first-to-last deltas across strata, never a
magnitude/scale-specific threshold, so it's already target-scale-agnostic.
Only the marginal/stratified table builders needed an xG variant (mean_xg
instead of rate, xg_sum instead of shots, no *100).

Usage:
    python -m src.eda.generate_confound_analysis_xg
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import PASSIVE
from src.eda.generate_confound_analysis import TEST_1, TEST_2, QCUT_N, _verdict

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "CONFOUND_ANALYSIS.json"

TARGET_XG = "target_future_xg_10s"


def _marginal_table_xg(df: pd.DataFrame, col: str, labels: list[str]) -> tuple[pd.Series, list[dict]]:
    bins = pd.qcut(df[col], QCUT_N, labels=labels)
    grouped = df.groupby(bins, observed=True)[TARGET_XG].agg(n="count", xg_sum="sum", mean_xg="mean")
    table = [
        # "rate" key name kept for compatibility with the reused _verdict(),
        # which hardcodes that key -- holds mean_xg here, not a shot-rate percentage.
        {"bin": str(idx), "n": int(row["n"]), "xg_sum": round(float(row["xg_sum"]), 3), "rate": round(float(row["mean_xg"]), 6)}
        for idx, row in grouped.iterrows()
    ]
    return bins, table


def _stratified_table_xg(df: pd.DataFrame, marginal_bins: pd.Series, confound_bins: pd.Series, marginal_labels: list[str], confound_labels: list[str]) -> list[dict]:
    grouped = df.groupby([confound_bins, marginal_bins], observed=True)[TARGET_XG].agg(n="count", xg_sum="sum", mean_xg="mean")
    strata = []
    for stratum_label in confound_labels:
        row_bins = []
        monotonic_rates = []
        for bin_label in marginal_labels:
            key = (stratum_label, bin_label)
            if key in grouped.index:
                r = grouped.loc[key]
                n, xg_sum, rate = int(r["n"]), round(float(r["xg_sum"]), 3), round(float(r["mean_xg"]), 6)
            else:
                n, xg_sum, rate = 0, 0.0, None
            row_bins.append({"bin": bin_label, "n": n, "xg_sum": xg_sum, "rate": rate})
            if rate is not None:
                monotonic_rates.append(rate)
        first_to_last_delta = (
            round(monotonic_rates[-1] - monotonic_rates[0], 6) if len(monotonic_rates) >= 2 else None
        )
        strata.append({
            "stratum": stratum_label,
            "bins": row_bins,
            "first_to_last_delta_pp": first_to_last_delta,
        })
    return strata


def run_test_xg(df: pd.DataFrame, spec: dict) -> dict:
    marginal_bins, marginal_table = _marginal_table_xg(df, spec["marginal_col"], spec["marginal_labels"])
    confound_bins = pd.qcut(df[spec["confound_col"]], QCUT_N, labels=spec["confound_labels"])
    strata = _stratified_table_xg(df, marginal_bins, confound_bins, spec["marginal_labels"], spec["confound_labels"])
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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])

    result_1 = run_test_xg(df, TEST_1)
    result_2 = run_test_xg(df, TEST_2)

    output = {
        "dataset": "passive",
        "parquet_path": PASSIVE["parquet_path"],
        "n_rows": len(df),
        "target_col": TARGET_XG,
        "methodology": (
            "xG counterpart of CONFOUND_ANALYSIS.json -- same quartile-stratification method, against mean xG "
            "instead of shot-rate percentage. _verdict()'s sign-only comparison is target-scale-agnostic, reused "
            "unchanged; only the table builders (mean_xg, not *100 rate) differ."
        ),
        "tests": [result_1, result_2],
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    for result in (result_1, result_2):
        print(f"\n=== {result['title']} (xG) ===")
        print("Marginal:", [(b["bin"], b["rate"]) for b in result["marginal_table"]])
        print(f"Verdict: {result['verdict']['verdict']} -- {result['verdict']['verdict_meaning']}")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
