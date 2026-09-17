"""CLI entrypoint: xG counterpart to generate_leakage_audit.py.

Part A (known-leaky-column scan, naming-pattern scan) is entirely
target-independent -- reused via import, not recomputed differently, since
it never groups by target at all.

Parts B/C/D quantify feature-vs-target splits and are inherently
target-dependent -- redone here against target_future_xg_10s: groupby mean
xG instead of shot-rate percentage, and a Welch's t-test (continuous
outcome, two groups) instead of a chi-square test of independence
(categorical outcome), since chi2_contingency requires a categorical
target and xG is continuous.

Usage:
    python -m src.eda.generate_leakage_audit_xg
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind

from src.eda.feature_config import PASSIVE
from src.eda.generate_leakage_audit import _candidate_columns, part_a

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "LEAKAGE_AUDIT.json"

TARGET_XG = "target_future_xg_10s"


def part_b_xg(df: pd.DataFrame) -> dict:
    grp = df.groupby("screened_option_was_avoided")[TARGET_XG].agg(n="count", mean_xg="mean")
    rows = {str(idx): {"n": int(r["n"]), "mean_xg": round(float(r["mean_xg"]), 6)} for idx, r in grp.iterrows()}
    xg_true = df.loc[df["screened_option_was_avoided"] == True, TARGET_XG]  # noqa: E712
    xg_false = df.loc[df["screened_option_was_avoided"] == False, TARGET_XG]  # noqa: E712
    t_stat, p_value = ttest_ind(xg_true, xg_false, equal_var=False)
    mean_true = rows.get("True", {}).get("mean_xg")
    mean_false = rows.get("False", {}).get("mean_xg")
    delta = round(mean_true - mean_false, 6) if mean_true is not None and mean_false is not None else None
    return {
        "column": "screened_option_was_avoided",
        "target_col": TARGET_XG,
        "by_value": rows,
        "delta": delta,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "verdict": "kept-excluded-conceptually; empirically near-flat xG difference, confirms low leakage risk for this specific pair",
        "note": "Excluded from the candidate list on conceptual/structural grounds regardless of this empirical result -- informational only.",
    }


def part_c_xg(df: pd.DataFrame) -> dict:
    grp = df.groupby("has_screened_outcome")[TARGET_XG].agg(n="count", mean_xg="mean")
    rows = {str(idx): {"n": int(r["n"]), "mean_xg": round(float(r["mean_xg"]), 6)} for idx, r in grp.iterrows()}
    xg_true = df.loc[df["has_screened_outcome"] == True, TARGET_XG]  # noqa: E712
    xg_false = df.loc[df["has_screened_outcome"] == False, TARGET_XG]  # noqa: E712
    t_stat, p_value = ttest_ind(xg_true, xg_false, equal_var=False)
    mean_true = rows.get("True", {}).get("mean_xg")
    mean_false = rows.get("False", {}).get("mean_xg")
    ratio = round(mean_true / mean_false, 2) if mean_false else None
    return {
        "column": "has_screened_outcome",
        "target_col": TARGET_XG,
        "by_value": rows,
        "ratio_true_to_false": ratio,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "verdict": "excluded -- censoring-mechanism proxy (structural exclusion holds; empirical strength differs from the binary-target audit)",
        "explanation": (
            f"Same censoring mechanism as the binary-target audit (False rows have their forward window "
            f"truncated by match/period end), but MUCH weaker empirically against xG: only a {ratio}x lower "
            f"mean xG ({mean_false} vs {mean_true}, Welch's t-test p={p_value:.3e}) -- barely significant, "
            "versus the binary target's ~12x ratio at p=1.5e-31. The exclusion still holds on structural grounds "
            "(the mechanism doesn't depend on which target is used), but don't cite this xG number as equally "
            "strong evidence -- it isn't. Plausible reason: xG is heavily right-skewed and this group has only "
            "n=2,860 rows, so the mean is noisier and less separated than the binary rate was."
        ),
    }


def part_d_xg(df: pd.DataFrame) -> dict:
    results = {}
    for col in ("has_option_2", "has_option_3"):
        grp = df.groupby(col)[TARGET_XG].agg(n="count", mean_xg="mean")
        rows = {str(idx): {"n": int(r["n"]), "mean_xg": round(float(r["mean_xg"]), 6)} for idx, r in grp.iterrows()}
        mean_true = rows.get("True", {}).get("mean_xg")
        mean_false = rows.get("False", {}).get("mean_xg")
        ratio = round(mean_false / mean_true, 2) if mean_true else None
        results[col] = {
            "column": col,
            "target_col": TARGET_XG,
            "by_value": rows,
            "ratio_false_to_true": ratio,
            "verdict": "flagged-for-follow-up",
            "explanation": (
                f"{col}=False shows a {ratio}x higher mean xG than True -- same direction as the binary-target "
                "audit. About the current freeze frame, not future-window computability; kept as a candidate "
                "feature, flagged for the same follow-up confound testing."
            ),
        }
    return results


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])
    candidate_cols = _candidate_columns(PASSIVE)

    a = part_a(df, candidate_cols)
    b = part_b_xg(df)
    c = part_c_xg(df)
    d = part_d_xg(df)

    output = {
        "dataset": "passive",
        "parquet_path": PASSIVE["parquet_path"],
        "n_rows": len(df),
        "target_col": TARGET_XG,
        "methodology": (
            "Part A (target-independent schema/naming scan) is reused unchanged from the binary-target audit. "
            "Parts B-D are redone against target_future_xg_10s: groupby mean xG instead of shot-rate percentage, "
            "Welch's t-test instead of chi-square (xG is continuous, not categorical)."
        ),
        "part_a_known_leaky_scan": a,
        "part_b_screened_option_was_avoided": b,
        "part_c_has_screened_outcome": c,
        "part_d_has_option_2_3": d,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("=== Part A ===", "clean" if a["verdict"] == "clean" else "ISSUES FOUND")
    print(f"Part C: has_screened_outcome ratio_true_to_false={c['ratio_true_to_false']} p={c['p_value']:.3e}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
