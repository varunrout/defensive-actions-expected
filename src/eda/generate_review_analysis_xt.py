"""CLI entrypoint: resolve the ACTIVE dataset's REVIEW-tier pairs from
CORRELATION_ANALYSIS_V1_HISTORICAL.json using xT-delta lift as evidence --
the xT sibling of generate_review_analysis_xg.py (the confirmed generator
of reports/analysis/xg_target/REVIEW_ANALYSIS.json / REVIEW_METHODOLOGY.html).

Exactly as in the xg version, only Type 2 (boolean vs boolean) actually
uses target evidence. Types 1 (structural correlation threshold), 3
(conceptual eta threshold) and 4 (name-pattern persistence check) never
reference a target at all, so their resolvers are IMPORTED AND REUSED
UNCHANGED from generate_review_analysis -- not reimplemented.

Adaptation 1 applies to the two Type-2 thresholds: the xg version uses
0.5x / 1.5x the dataset's overall mean xg. Those are translated here via
xg's own std-fraction onto target_xt_delta's standard deviation (see
src/eda/xt_common.py). Adaptation 3 applies to the opposite-sign rule,
which is if anything MORE meaningful on this target: a sign disagreement
here means one flag marks threat-reducing actions and the other
threat-increasing, which is a substantive football disagreement rather
than a difference in degree.

Active-binary leg only; the passive half of the xg file is out of scope.

Usage:
    python -m src.eda.generate_review_analysis_xt
"""

from __future__ import annotations

import json

import pandas as pd

from src.eda import xt_common as xc
from src.eda.generate_review_analysis import (
    TYPE1_R_THRESHOLD,
    TYPE2_SUBSET_CONTAINMENT,
    _containment,
    _dropped_columns_for_dataset,
    _is_persistence_pair,
    _resolve_type1,
    _resolve_type3,
)

REPO_ROOT = xc.REPO_ROOT
CORRELATION_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CORRELATION_ANALYSIS_V1_HISTORICAL.json"
OUT_DIR = xc.OUT_DIR
OUTPUT_PATH = OUT_DIR / "REVIEW_ANALYSIS.json"
TARGET = xc.TARGET


def _resolve_type2_xt(pair: dict, df: pd.DataFrame, lifts: dict[str, dict], scale: dict) -> dict:
    col_a, col_b = pair["feature_a"], pair["feature_b"]
    lift_a, lift_b = lifts[col_a]["lift"], lifts[col_b]["lift"]
    pct_a, pct_b = lifts[col_a]["pct_true"], lifts[col_b]["pct_true"]
    same_sign = (lift_a >= 0) == (lift_b >= 0)
    diff = abs(lift_a - lift_b)
    near_identical = scale["near_identical_threshold"]
    distinct_signal = scale["distinct_signal_threshold"]

    evidence = {
        "lift_a": round(lift_a, 8),
        "lift_b": round(lift_b, 8),
        "pct_true_a": round(pct_a, 2),
        "pct_true_b": round(pct_b, 2),
        "lift_diff": round(diff, 8),
        "same_sign": same_sign,
    }

    if not same_sign:
        return {
            "relationship_type": 2,
            "verdict": "keep_both",
            "evidence": evidence,
            "reason": (
                f"Opposite-sign xT-delta lift ({lift_a:+.6f} vs {lift_b:+.6f}) -- the two flags disagree about "
                "the DIRECTION of the threat swing, not just its size: one marks rows where xT falls across the "
                "action, the other rows where it rises. On a symmetric, sign-carrying target that is a "
                "substantive football disagreement, and real signal."
            ),
        }

    p_b_given_a, p_a_given_b = _containment(df, col_a, col_b)
    evidence["containment_b_given_a"] = round(p_b_given_a, 4)
    evidence["containment_a_given_b"] = round(p_a_given_b, 4)

    if p_b_given_a >= TYPE2_SUBSET_CONTAINMENT or p_a_given_b >= TYPE2_SUBSET_CONTAINMENT:
        subset_col, superset_col = (col_a, col_b) if p_b_given_a >= TYPE2_SUBSET_CONTAINMENT else (col_b, col_a)
        keep_col, drop_col = (col_a, col_b) if abs(lift_a) >= abs(lift_b) else (col_b, col_a)
        return {
            "relationship_type": 2,
            "verdict": "drop_a" if drop_col == col_a else "drop_b",
            "evidence": evidence,
            "reason": (
                f"{subset_col} is a near-total subset of {superset_col} (containment >= {TYPE2_SUBSET_CONTAINMENT:.0%}); "
                f"same sign, so keep the flag with the larger |lift| in EITHER direction ({keep_col}, lift "
                f"{lifts[keep_col]['lift']:+.6f}) and drop the other ({drop_col}, lift {lifts[drop_col]['lift']:+.6f})."
            ),
        }

    if diff < near_identical:
        broader_col, narrower_col = (col_a, col_b) if pct_a >= pct_b else (col_b, col_a)
        return {
            "relationship_type": 2,
            "verdict": "drop_a" if broader_col == col_a else "drop_b",
            "evidence": evidence,
            "reason": (
                f"Near-identical xT-delta lift ({lift_a:+.6f} vs {lift_b:+.6f}, diff {diff:.6f} < "
                f"{near_identical:.6f}), same signal twice -- drop the broader flag ({broader_col}, "
                f"{lifts[broader_col]['pct_true']:.1f}% True), keep the narrower one ({narrower_col})."
            ),
        }

    if diff >= distinct_signal:
        return {
            "relationship_type": 2,
            "verdict": "keep_both",
            "evidence": evidence,
            "reason": (
                f"Same sign but xT-delta lift diverges by {diff:.6f} (>= {distinct_signal:.6f}) with no subset "
                "relationship -- both carry distinct signal."
            ),
        }

    return {
        "relationship_type": 2,
        "verdict": "needs_human_call",
        "evidence": evidence,
        "reason": (
            f"Ambiguous zone: same sign, xT-delta lift diff {diff:.6f} is between {near_identical:.6f} and "
            f"{distinct_signal:.6f}, and no clear subset relationship -- not forcing a verdict."
        ),
    }


def resolve_active_xt(correlation_data: dict) -> dict:
    correlation_ds = correlation_data["datasets"]["active"]
    review_pairs = correlation_ds["pairs"]["review"]
    dropped_columns = _dropped_columns_for_dataset(correlation_ds)

    df = xc.load_active_xt()
    scale = xc.target_scale(df)

    boolean_cols = sorted(
        {p["feature_a"] for p in review_pairs if p["type_a"] == "boolean"}
        | {p["feature_b"] for p in review_pairs if p["type_b"] == "boolean"}
    )
    lifts = {c: xc.boolean_xt_lift(df, c) for c in boolean_cols}

    resolved: list[dict] = []
    type1_boolean_verdict: dict[str, dict] = {}
    deferred: list[dict] = []

    for pair in review_pairs:
        types = {pair["type_a"], pair["type_b"]}

        if _is_persistence_pair(pair["feature_a"], pair["feature_b"]):
            resolved.append({**pair, "relationship_type": 4, "verdict": "keep_both",
                             "reason": "State-persistence / autocorrelation pair -- not redundancy "
                                       "(target-independent check, reused unchanged)."})
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
            result = _resolve_type2_xt(pair, df, lifts, scale)
            resolved.append({**pair, **result})
            if result["verdict"].startswith("drop_"):
                drop_col = pair["feature_a"] if result["verdict"] == "drop_a" else pair["feature_b"]
                type1_boolean_verdict.setdefault(drop_col, {"verdict": "drop_boolean", "reason": result["reason"]})
            continue

        if types == {"categorical", "numeric"}:
            resolved.append({**pair, **_resolve_type3(pair)})
            continue

        if types == {"numeric"}:
            resolved.append({**pair, "relationship_type": None, "verdict": "needs_human_call",
                             "reason": "Continuous<->continuous pair; no automated rule covers this combination "
                                       "(target-independent -- same as the binary- and xg-target reviews)."})
            continue

        raise ValueError(f"Unhandled REVIEW pair type combination: {types}")

    for pair in deferred:
        bool_col = pair["feature_a"] if pair["type_a"] == "boolean" else pair["feature_b"]
        prior = type1_boolean_verdict.get(bool_col)
        if prior is not None:
            resolved.append({**pair, "relationship_type": 1, "verdict": "superseded",
                             "reason": f"{bool_col} already resolved for removal by a separate, stronger "
                                       f"pairing ({prior['reason']})."})
        else:
            resolved.append({**pair, "relationship_type": None, "verdict": "needs_human_call",
                             "reason": f"Boolean-continuous pair below the Type-1 structural threshold "
                                       f"(|r|={pair['abs_value']} < {TYPE1_R_THRESHOLD}); no automated rule applies."})

    needs_human_call = [r for r in resolved if r["verdict"] == "needs_human_call"]
    n_opposite_sign = sum(
        1 for r in resolved
        if r.get("relationship_type") == 2 and r.get("evidence", {}).get("same_sign") is False
    )

    return {
        "dataset": "active",
        "target": TARGET,
        "target_scale": scale,
        "near_identical_threshold": scale["near_identical_threshold"],
        "distinct_signal_threshold": scale["distinct_signal_threshold"],
        "n_review_pairs": len(review_pairs),
        "n_type2_opposite_sign_pairs": n_opposite_sign,
        "resolved_pairs": resolved,
        "needs_human_call": needs_human_call,
        "verdict_counts": pd.Series([r["verdict"] for r in resolved]).value_counts().to_dict(),
        "lifts": lifts,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    correlation_data = json.loads(CORRELATION_PATH.read_text(encoding="utf-8"))
    active = resolve_active_xt(correlation_data)

    output = {
        "generated_at": correlation_data["generated_at"],
        "source": str(CORRELATION_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "leg": "active-binary only (passive leg out of scope for this target)",
        "note": (
            "xT-delta-lift counterpart of reports/analysis/xg_target/REVIEW_ANALYSIS.json, ACTIVE half only. "
            "Only Type 2 verdicts can differ from the binary- and xg-target versions; Types 1/3/4 are "
            "target-independent and their resolvers are imported and reused unchanged. The two Type-2 "
            "thresholds are translated from the xg suite's 0.5x/1.5x-mean-xg convention via xg's own "
            "standard-deviation fraction -- a multiple of target_xt_delta's own near-zero, NEGATIVE mean would "
            "be meaningless. The opposite-sign rule carries more weight on this target than on xg: a sign "
            "disagreement means the two flags disagree about the DIRECTION of the threat swing."
        ),
        "datasets": {"active": active},
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print(f"active: {active['n_review_pairs']} review pairs, {len(active['needs_human_call'])} needs_human_call, "
          f"{active['n_type2_opposite_sign_pairs']} Type-2 opposite-sign pair(s)")
    print(f"  thresholds: near-identical {active['near_identical_threshold']:.6f}, "
          f"distinct-signal {active['distinct_signal_threshold']:.6f}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
