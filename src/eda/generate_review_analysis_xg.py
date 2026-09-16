"""CLI entrypoint: resolve REVIEW-tier pairs from
CORRELATION_ANALYSIS_V1_HISTORICAL.json using xG lift instead of shot-rate
lift as evidence -- the xG counterpart to generate_review_analysis.py /
REVIEW_ANALYSIS_V1_HISTORICAL.json.

Of the four relationship types, only Type 2 (boolean vs boolean) actually
uses target evidence (lift comparison) -- Type 1 is a structural
correlation threshold, Type 3 is a conceptual eta threshold, and Type 4 is a
name-pattern check. None of those three change with the target, so their
resolution logic is imported and reused unchanged; only Type 2 gets an xG
variant here.

xG lift thresholds are RELATIVE to each dataset's own overall mean xG, not
fixed percentage-point margins (same reasoning as
generate_numerical_xg_target_analysis.py -- xG is heavily zero-inflated,
mean ~0.006-0.008, nothing like a 0-100% rate):
  NEAR_IDENTICAL_RATIO = 0.5 (lift diff < ratio * overall_mean_xg -> same signal twice)
  DISTINCT_SIGNAL_RATIO = 1.5 (lift diff >= ratio * overall_mean_xg -> distinct signal)
Both stated explicitly in the JSON output, not just implied.

Usage:
    python -m src.eda.generate_review_analysis_xg
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda import compute_stats_xg as cs_xg
from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_review_analysis import (
    TYPE1_R_THRESHOLD,
    TYPE2_SUBSET_CONTAINMENT,
    TYPE3_ETA_FLAG_THRESHOLD,
    _containment,
    _dropped_columns_for_dataset,
    _is_persistence_pair,
    _resolve_type1,
    _resolve_type3,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CORRELATION_PATH = REPO_ROOT / "reports" / "eda" / "CORRELATION_ANALYSIS_V1_HISTORICAL.json"
OUT_DIR = REPO_ROOT / "reports" / "eda_xg"
OUTPUT_PATH = OUT_DIR / "REVIEW_ANALYSIS.json"
TARGET_XG = "target_future_xg_10s"

NEAR_IDENTICAL_RATIO = 0.5
DISTINCT_SIGNAL_RATIO = 1.5


def _load_xg_lifts(df: pd.DataFrame, boolean_cols: list[str]) -> dict[str, dict]:
    return {col: cs_xg.boolean_xg_lift(df, col, TARGET_XG) for col in boolean_cols}


def _resolve_type2_xg(pair: dict, df: pd.DataFrame, lifts: dict[str, dict], overall_mean_xg: float) -> dict:
    col_a, col_b = pair["feature_a"], pair["feature_b"]
    lift_a, lift_b = lifts[col_a]["lift"], lifts[col_b]["lift"]
    pct_a, pct_b = lifts[col_a]["pct_true"], lifts[col_b]["pct_true"]
    same_sign = (lift_a >= 0) == (lift_b >= 0)
    diff = abs(lift_a - lift_b)
    near_identical = NEAR_IDENTICAL_RATIO * overall_mean_xg
    distinct_signal = DISTINCT_SIGNAL_RATIO * overall_mean_xg

    evidence = {
        "lift_a": round(lift_a, 6),
        "lift_b": round(lift_b, 6),
        "pct_true_a": round(pct_a, 2),
        "pct_true_b": round(pct_b, 2),
        "lift_diff": round(diff, 6),
        "same_sign": same_sign,
    }

    if not same_sign:
        return {
            "relationship_type": 2,
            "verdict": "keep_both",
            "evidence": evidence,
            "reason": f"Opposite-sign xG lift ({lift_a:+.5f} vs {lift_b:+.5f}) -- the two flags disagree, which is real signal.",
        }

    p_b_given_a, p_a_given_b = _containment(df, col_a, col_b)
    evidence["containment_b_given_a"] = round(p_b_given_a, 4)
    evidence["containment_a_given_b"] = round(p_a_given_b, 4)

    if p_b_given_a >= TYPE2_SUBSET_CONTAINMENT or p_a_given_b >= TYPE2_SUBSET_CONTAINMENT:
        subset_col, superset_col = (col_a, col_b) if p_b_given_a >= TYPE2_SUBSET_CONTAINMENT else (col_b, col_a)
        keep_col, drop_col = (col_a, col_b) if abs(lift_a) >= abs(lift_b) else (col_b, col_a)
        verdict = "drop_a" if drop_col == col_a else "drop_b"
        return {
            "relationship_type": 2,
            "verdict": verdict,
            "evidence": evidence,
            "reason": (
                f"{subset_col} is a near-total subset of {superset_col} (containment >= {TYPE2_SUBSET_CONTAINMENT:.0%}); "
                f"same sign, so keep the higher-xG-signal flag ({keep_col}, lift {lifts[keep_col]['lift']:+.5f}) and "
                f"drop the other ({drop_col}, lift {lifts[drop_col]['lift']:+.5f})."
            ),
        }

    if diff < near_identical:
        broader_col, narrower_col = (col_a, col_b) if pct_a >= pct_b else (col_b, col_a)
        verdict = "drop_a" if broader_col == col_a else "drop_b"
        return {
            "relationship_type": 2,
            "verdict": verdict,
            "evidence": evidence,
            "reason": (
                f"Near-identical xG lift ({lift_a:+.5f} vs {lift_b:+.5f}, diff {diff:.5f} < {near_identical:.5f} "
                f"= {NEAR_IDENTICAL_RATIO}x overall mean xG), same signal twice -- drop the broader flag "
                f"({broader_col}, {lifts[broader_col]['pct_true']:.1f}% True), keep the narrower one ({narrower_col})."
            ),
        }

    if diff >= distinct_signal:
        return {
            "relationship_type": 2,
            "verdict": "keep_both",
            "evidence": evidence,
            "reason": (
                f"Same sign but xG lift diverges by {diff:.5f} (>= {distinct_signal:.5f} = {DISTINCT_SIGNAL_RATIO}x "
                "overall mean xG) with no subset relationship -- both carry distinct signal."
            ),
        }

    return {
        "relationship_type": 2,
        "verdict": "needs_human_call",
        "evidence": evidence,
        "reason": (
            f"Ambiguous zone: same sign, xG lift diff {diff:.5f} is between {near_identical:.5f} and "
            f"{distinct_signal:.5f}, and no clear subset relationship -- not forcing a verdict."
        ),
    }


def resolve_dataset_xg(dataset_key: str, correlation_data: dict) -> dict:
    dataset_cfg = {"active": ACTIVE, "passive": PASSIVE}[dataset_key]
    correlation_ds = correlation_data["datasets"][dataset_key]
    review_pairs = correlation_ds["pairs"]["review"]
    dropped_columns = _dropped_columns_for_dataset(correlation_ds)

    df = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"])
    overall_mean_xg = float(df[TARGET_XG].mean())

    # boolean list for lifts: every boolean column that appears in any review
    # pair (reconstructed pool may include dropped booleans not in the
    # locked list -- collect from the pairs themselves, not dataset_cfg).
    boolean_cols = sorted(
        {p["feature_a"] for p in review_pairs if p["type_a"] == "boolean"}
        | {p["feature_b"] for p in review_pairs if p["type_b"] == "boolean"}
    )
    lifts = _load_xg_lifts(df, boolean_cols) if boolean_cols else {}

    resolved: list[dict] = []
    type1_boolean_verdict: dict[str, dict] = {}
    deferred: list[dict] = []

    for pair in review_pairs:
        types = {pair["type_a"], pair["type_b"]}

        if _is_persistence_pair(pair["feature_a"], pair["feature_b"]):
            resolved.append({
                **pair,
                "relationship_type": 4,
                "verdict": "keep_both",
                "reason": "State-persistence / autocorrelation pair -- not redundancy (target-independent check).",
            })
            continue

        if types == {"boolean", "numeric"}:
            if pair["abs_value"] >= TYPE1_R_THRESHOLD:
                result = _resolve_type1(pair, dropped_columns)
                resolved.append({**pair, **result})
                if result["verdict"] in ("drop_boolean", "moot_after_drop"):
                    type1_boolean_verdict[result["boolean_column"]] = result
            else:
                deferred.append(pair)
            continue

        if types == {"boolean"}:
            result = _resolve_type2_xg(pair, df, lifts, overall_mean_xg)
            resolved.append({**pair, **result})
            if result["verdict"].startswith("drop_"):
                drop_col = pair["feature_a"] if result["verdict"] == "drop_a" else pair["feature_b"]
                type1_boolean_verdict.setdefault(drop_col, {"verdict": "drop_boolean", "reason": result["reason"]})
            continue

        if types == {"categorical", "numeric"}:
            result = _resolve_type3(pair)
            resolved.append({**pair, **result})
            continue

        if types == {"numeric"}:
            resolved.append({
                **pair,
                "relationship_type": None,
                "verdict": "needs_human_call",
                "reason": (
                    "Continuous<->continuous pair; no automated rule covers this combination "
                    "(target-independent -- same as the binary-target review)."
                ),
            })
            continue

        raise ValueError(f"Unhandled REVIEW pair type combination: {types} ({pair['feature_a']}/{pair['feature_b']})")

    for pair in deferred:
        bool_col = pair["feature_a"] if pair["type_a"] == "boolean" else pair["feature_b"]
        prior = type1_boolean_verdict.get(bool_col)
        if prior is not None:
            resolved.append({
                **pair,
                "relationship_type": 1,
                "verdict": "superseded",
                "reason": f"{bool_col} already resolved for removal by a separate, stronger pairing ({prior['reason']}).",
            })
        else:
            resolved.append({
                **pair,
                "relationship_type": None,
                "verdict": "needs_human_call",
                "reason": (
                    f"Boolean-continuous pair below the Type-1 structural threshold (|r|={pair['abs_value']} < "
                    f"{TYPE1_R_THRESHOLD}); no automated rule applies."
                ),
            })

    needs_human_call = [r for r in resolved if r["verdict"] == "needs_human_call"]

    return {
        "dataset": dataset_key,
        "target": TARGET_XG,
        "overall_mean_xg": round(overall_mean_xg, 6),
        "near_identical_threshold": round(NEAR_IDENTICAL_RATIO * overall_mean_xg, 6),
        "distinct_signal_threshold": round(DISTINCT_SIGNAL_RATIO * overall_mean_xg, 6),
        "n_review_pairs": len(review_pairs),
        "resolved_pairs": resolved,
        "needs_human_call": needs_human_call,
        "verdict_counts": pd.Series([r["verdict"] for r in resolved]).value_counts().to_dict(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    correlation_data = json.loads(CORRELATION_PATH.read_text(encoding="utf-8"))

    results = {key: resolve_dataset_xg(key, correlation_data) for key in ("active", "passive")}

    output = {
        "generated_at": correlation_data["generated_at"],
        "source": str(CORRELATION_PATH.relative_to(REPO_ROOT)),
        "note": "xG-lift counterpart of REVIEW_ANALYSIS_V1_HISTORICAL.json -- only Type 2 verdicts can differ from the binary-target version; Types 1/3/4 are target-independent and reused unchanged.",
        "datasets": results,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("\n=== needs_human_call (xG) ===")
    for key, r in results.items():
        print(f"\n{key}: {len(r['needs_human_call'])} pair(s)  (overall mean xG={r['overall_mean_xg']})")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
