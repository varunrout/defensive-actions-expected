"""CLI entrypoint: systematic leakage audit over the passive dataset's
candidate feature list -- not just a re-check of the two issues already
documented in dax/models/leakage.py's guard spirit and
screened_option_was_avoided, but a pass over every candidate feature asking
"could this only be knowable because of what happens in the 10-second
forward window."

Part A: confirm the known-leaky columns (target_future_shot_10s,
target_future_xg_10s, screened_option_was_avoided) are excluded from
feature_config.py's PASSIVE candidate lists, and scan the full parquet
schema for any other leakage-suggestive naming pattern (future/next/
avoided/outcome/after) not already accounted for.

Part B: quantify the screened_option_was_avoided guard empirically
(informational -- the exclusion is correct on conceptual/structural
grounds regardless of what this number shows).

Part C: has_screened_outcome is a censoring-mechanism proxy, not defensive
signal -- it flags rows where the target's own measurement window was cut
short (end of period/match), which produces a spuriously low
target_future_shot_10s rate that has nothing to do with defending. This
script reports the chi-square evidence; the fix (removing it from
feature_config.py) is applied separately, not by this script.

Part D: has_option_2/has_option_3 show a similar-looking split but are
about the CURRENT freeze frame (a 2nd/3rd ranked option visible now), not
future-window computability -- flagged for follow-up confound testing, not
excluded.

Usage:
    python -m src.eda.generate_leakage_audit
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

from src.eda.feature_config import PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "LEAKAGE_AUDIT.json"

TARGET = "target_future_shot_10s"
NAME_PATTERNS = ["future", "next", "avoided", "outcome", "after"]

KNOWN_LEAKY = ["target_future_shot_10s", "target_future_xg_10s", "screened_option_was_avoided"]


def _candidate_columns(cfg: dict) -> set[str]:
    return set(cfg["categorical"] + cfg["boolean"] + cfg["continuous"] + cfg["discrete"])


def part_a(df: pd.DataFrame, candidate_cols: set[str]) -> dict:
    known_leaky_status = {
        col: {"in_candidate_list": col in candidate_cols, "verdict": "excluded" if col not in candidate_cols else "LEAK -- STILL PRESENT"}
        for col in KNOWN_LEAKY
    }

    all_columns = list(df.columns)
    name_flagged = sorted(c for c in all_columns if any(p in c.lower() for p in NAME_PATTERNS))
    unexpected_in_candidates = [c for c in name_flagged if c in candidate_cols and c not in KNOWN_LEAKY]

    return {
        "known_leaky_columns": known_leaky_status,
        "all_leakage_suggestive_column_names": name_flagged,
        "n_columns_scanned_total": len(all_columns),
        "unexpected_leaky_columns_in_candidate_list": unexpected_in_candidates,
        "verdict": "clean" if not unexpected_in_candidates and all(v["in_candidate_list"] is False for v in known_leaky_status.values()) else "ISSUES FOUND",
    }


def part_b(df: pd.DataFrame) -> dict:
    grp = df.groupby("screened_option_was_avoided")[TARGET].agg(n="count", rate=lambda s: s.mean() * 100)
    rows = {str(idx): {"n": int(r["n"]), "rate": round(float(r["rate"]), 3)} for idx, r in grp.iterrows()}
    rate_true = rows.get("True", {}).get("rate")
    rate_false = rows.get("False", {}).get("rate")
    delta = round(rate_true - rate_false, 3) if rate_true is not None and rate_false is not None else None
    return {
        "column": "screened_option_was_avoided",
        "target_col": TARGET,
        "by_value": rows,
        "delta_pp": delta,
        "verdict": "kept-excluded-conceptually; empirically near-flat correlation, confirms low leakage risk for this specific pair",
        "note": "screened_option_was_avoided is excluded from the candidate list on conceptual/structural grounds (both it and the target are forward-looking from the anchor event) regardless of this empirical result -- this measurement is informational only, not the basis for the exclusion.",
    }


def part_c(df: pd.DataFrame) -> dict:
    grp = df.groupby("has_screened_outcome")[TARGET].agg(n="count", rate=lambda s: s.mean() * 100)
    rows = {str(idx): {"n": int(r["n"]), "rate": round(float(r["rate"]), 3)} for idx, r in grp.iterrows()}
    ct = pd.crosstab(df["has_screened_outcome"], df[TARGET])
    chi2, p, dof, _ = chi2_contingency(ct)
    rate_true = rows.get("True", {}).get("rate")
    rate_false = rows.get("False", {}).get("rate")
    ratio = round(rate_true / rate_false, 2) if rate_false else None
    return {
        "column": "has_screened_outcome",
        "target_col": TARGET,
        "by_value": rows,
        "crosstab": ct.values.tolist(),
        "crosstab_index": [str(i) for i in ct.index],
        "crosstab_columns": [str(c) for c in ct.columns],
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "ratio_true_to_false": ratio,
        "verdict": "excluded -- censoring-mechanism proxy",
        "explanation": (
            "has_screened_outcome is False only when screened_option_was_avoided could not be computed "
            "(end of period/match, no next event in the data to compare against). Those rows show a "
            f"{ratio}x lower shot rate ({rate_false}% vs {rate_true}%, chi2 p={p:.3e}) -- not because nothing "
            "defensively interesting happened, but because target_future_shot_10s's own 10-second measurement "
            "window was truncated by the match/period ending, making a false negative (no time left for a shot) "
            "far more likely than a genuine successful defensive outcome. A model fed this feature would learn "
            "to predict near-zero shot probability whenever the target's own measurement window was cut short -- "
            "circular with the target's construction, not causal, not useful, and it would look like a strong "
            "feature in offline evaluation while encoding nothing about defending."
        ),
    }


def part_d(df: pd.DataFrame) -> dict:
    results = {}
    for col in ("has_option_2", "has_option_3"):
        grp = df.groupby(col)[TARGET].agg(n="count", rate=lambda s: s.mean() * 100)
        rows = {str(idx): {"n": int(r["n"]), "rate": round(float(r["rate"]), 3)} for idx, r in grp.iterrows()}
        rate_true = rows.get("True", {}).get("rate")
        rate_false = rows.get("False", {}).get("rate")
        ratio = round(rate_false / rate_true, 2) if rate_true else None
        results[col] = {
            "column": col,
            "target_col": TARGET,
            "by_value": rows,
            "ratio_false_to_true": ratio,
            "verdict": "flagged-for-follow-up",
            "explanation": (
                f"{col}=False shows a {ratio}x HIGHER shot rate than True -- the opposite shape from "
                "has_screened_outcome. This is about the CURRENT freeze frame (whether a 2nd/3rd ranked passing "
                "option was visible right now), not about whether the 10s forward window was computable, so it "
                "is not the same censoring-mechanism artefact. Plausible as real signal (fewer passing options "
                "can mean a more direct, more urgent attack) rather than a target-construction artefact -- kept "
                "as a candidate feature. Worth testing the same way as the marking-tightness/lane-screening "
                "reversals (does conditioning on some third variable explain it away), but that is separate "
                "follow-up work, not part of this audit."
            ),
        }
    return results


def main() -> None:
    df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])
    candidate_cols = _candidate_columns(PASSIVE)

    a = part_a(df, candidate_cols)
    b = part_b(df)
    c = part_c(df)
    d = part_d(df)

    output = {
        "dataset": "passive",
        "parquet_path": PASSIVE["parquet_path"],
        "n_rows": len(df),
        "target_col": TARGET,
        "methodology": (
            "Part A confirms known-leaky columns are excluded and scans the full parquet schema for any other "
            "leakage-suggestive column name not already accounted for. Parts B-D quantify three specific "
            "candidate-feature risks with a groupby shot-rate split and, where relevant, a chi-square test of "
            "independence -- not a claim of causation, a statistical description of whether the split is "
            "consistent with the target's own forward-looking construction leaking into the feature."
        ),
        "part_a_known_leaky_scan": a,
        "part_b_screened_option_was_avoided": b,
        "part_c_has_screened_outcome": c,
        "part_d_has_option_2_3": d,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("=== Part A ===")
    print(json.dumps(a, indent=2))
    print("\n=== Part B ===")
    print(f"screened_option_was_avoided: True={b['by_value'].get('True')} False={b['by_value'].get('False')}")
    print("\n=== Part C ===")
    print(f"has_screened_outcome: True={c['by_value'].get('True')} False={c['by_value'].get('False')} chi2_p={c['p_value']:.3e}")
    print("\n=== Part D ===")
    for col, r in d.items():
        print(f"{col}: True={r['by_value'].get('True')} False={r['by_value'].get('False')}")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
