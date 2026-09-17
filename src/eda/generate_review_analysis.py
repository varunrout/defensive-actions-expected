"""CLI entrypoint: resolve every REVIEW-tier pair from CORRELATION_ANALYSIS.json
into a final verdict. Correlation alone never decides a verdict here -- it
only got the pair onto the REVIEW list in prompt 1/4. This script uses actual
shot-rate lift (Type 2), a structural boolean-vs-parent test (Type 1), a
conceptual categorical-vs-continuous check (Type 3), or a naming-pattern
autocorrelation check (Type 4) as evidence.

Pairs that genuinely can't be resolved by these tests are surfaced as
`needs_human_call: true`, never force-resolved.

Usage:
    python -m src.eda.generate_review_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda import compute_stats as cs
from src.eda.feature_config import DATASETS

REPO_ROOT = Path(__file__).resolve().parents[2]
CORRELATION_PATH = REPO_ROOT / "reports" / "eda" / "CORRELATION_ANALYSIS.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "REVIEW_ANALYSIS.json"

TYPE1_R_THRESHOLD = 0.75
TYPE2_R_THRESHOLD = 0.5
TYPE2_NEAR_IDENTICAL_PP = 1.0
TYPE2_DISTINCT_SIGNAL_PP = 3.0
TYPE2_SUBSET_CONTAINMENT = 0.90
TYPE3_ETA_FLAG_THRESHOLD = 0.85

PERSISTENCE_SUFFIXES = ("_prev_event", "_prev", "_lag", "_before")

# Known DROP-tier resolutions (which side survives) -- sourced from the
# Cowork pass this pipeline replicates (see prompt 2/4's own known-resolution
# table and prompt 4/4 Part C). Used only to decide whether a Type-1
# boolean's continuous "parent" is itself already scheduled for removal
# elsewhere, in which case this pair becomes moot rather than independently
# actionable.
DROP_TIER_KEEP = {
    frozenset({"position", "position_group"}): "position",
    frozenset({"event_type", "action_family"}): "event_type",
    frozenset({"attacking_goal_centrality", "distance_to_center_line"}): "attacking_goal_centrality",
    frozenset({"distance_to_defending_goal", "zone_defensive_value"}): "distance_to_defending_goal",
}


def _is_persistence_pair(col_a: str, col_b: str) -> bool:
    def strip(col: str) -> str | None:
        for suf in PERSISTENCE_SUFFIXES:
            if col.endswith(suf):
                return col[: -len(suf)]
        return None

    base_a, base_b = strip(col_a), strip(col_b)
    if base_a is not None and base_a == col_b:
        return True
    if base_b is not None and base_b == col_a:
        return True
    return False


def _dropped_columns_for_dataset(correlation_ds: dict) -> set[str]:
    dropped: set[str] = set()
    for pair in correlation_ds["pairs"]["drop"]:
        key = frozenset({pair["feature_a"], pair["feature_b"]})
        keep = DROP_TIER_KEEP.get(key)
        if keep is None:
            raise ValueError(
                f"No known keep-preference for DROP pair {pair['feature_a']}/{pair['feature_b']} -- "
                "add one to DROP_TIER_KEEP in generate_review_analysis.py before trusting Type-1 output."
            )
        drop = pair["feature_b"] if keep == pair["feature_a"] else pair["feature_a"]
        dropped.add(drop)
    return dropped


def _load_lifts(df: pd.DataFrame, dataset_cfg: dict) -> dict[str, dict]:
    target = dataset_cfg["target_col"]
    return {col: cs.boolean_lift(df, col, target) for col in dataset_cfg["boolean"]}


def _containment(df: pd.DataFrame, col_a: str, col_b: str) -> tuple[float, float]:
    a = df[col_a].astype(bool)
    b = df[col_b].astype(bool)
    n_a, n_b = int(a.sum()), int(b.sum())
    both = int((a & b).sum())
    p_b_given_a = both / n_a if n_a else float("nan")
    p_a_given_b = both / n_b if n_b else float("nan")
    return p_b_given_a, p_a_given_b


def _resolve_type1(pair: dict, dropped_columns: set[str]) -> dict:
    bool_col, num_col = (pair["feature_a"], pair["feature_b"]) if pair["type_a"] == "boolean" else (pair["feature_b"], pair["feature_a"])
    if num_col in dropped_columns:
        keep_side = next(k for k, v in DROP_TIER_KEEP.items() if num_col in k and DROP_TIER_KEEP[k] != num_col)
        surviving = DROP_TIER_KEEP[keep_side]
        return {
            "relationship_type": 1,
            "verdict": "moot_after_drop",
            "boolean_column": bool_col,
            "continuous_column": num_col,
            "reason": (
                f"{num_col} is itself scheduled for removal in the DROP tier (redundant with {surviving}); "
                f"once that DROP is applied, this pair no longer exists. See the {surviving}-based pairing for "
                f"the same conclusion about {bool_col}."
            ),
        }
    return {
        "relationship_type": 1,
        "verdict": "drop_boolean",
        "boolean_column": bool_col,
        "continuous_column": num_col,
        "reason": (
            f"{bool_col} is structurally a threshold cut on {num_col} (|r|={pair['abs_value']} >= {TYPE1_R_THRESHOLD}); "
            f"drop the boolean, keep the continuous parent."
        ),
    }


def _resolve_type2(pair: dict, df: pd.DataFrame, lifts: dict[str, dict]) -> dict:
    col_a, col_b = pair["feature_a"], pair["feature_b"]
    lift_a, lift_b = lifts[col_a]["lift"], lifts[col_b]["lift"]
    pct_a, pct_b = lifts[col_a]["pct_true"], lifts[col_b]["pct_true"]
    same_sign = (lift_a >= 0) == (lift_b >= 0)
    diff = abs(lift_a - lift_b)

    evidence = {
        "lift_a": round(lift_a, 3),
        "lift_b": round(lift_b, 3),
        "pct_true_a": round(pct_a, 2),
        "pct_true_b": round(pct_b, 2),
        "lift_diff_pp": round(diff, 3),
        "same_sign": same_sign,
    }

    if not same_sign:
        return {
            "relationship_type": 2,
            "verdict": "keep_both",
            "evidence": evidence,
            "reason": f"Opposite-sign lift ({lift_a:+.2f}pp vs {lift_b:+.2f}pp) -- the two flags disagree, which is real signal.",
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
                f"same sign, so keep the higher-signal flag ({keep_col}, lift {lifts[keep_col]['lift']:+.2f}pp) and "
                f"drop the other ({drop_col}, lift {lifts[drop_col]['lift']:+.2f}pp)."
            ),
        }

    if diff < TYPE2_NEAR_IDENTICAL_PP:
        broader_col, narrower_col = (col_a, col_b) if pct_a >= pct_b else (col_b, col_a)
        verdict = "drop_a" if broader_col == col_a else "drop_b"
        return {
            "relationship_type": 2,
            "verdict": verdict,
            "evidence": evidence,
            "reason": (
                f"Near-identical lift ({lift_a:+.2f}pp vs {lift_b:+.2f}pp, diff {diff:.2f}pp < {TYPE2_NEAR_IDENTICAL_PP}pp), "
                f"same signal twice -- drop the broader flag ({broader_col}, {lifts[broader_col]['pct_true']:.1f}% True), "
                f"keep the narrower one ({narrower_col}, {lifts[narrower_col]['pct_true']:.1f}% True)."
            ),
        }

    if diff >= TYPE2_DISTINCT_SIGNAL_PP:
        return {
            "relationship_type": 2,
            "verdict": "keep_both",
            "evidence": evidence,
            "reason": f"Same sign but lift diverges by {diff:.2f}pp (>= {TYPE2_DISTINCT_SIGNAL_PP}pp) with no subset relationship -- both carry distinct signal.",
        }

    return {
        "relationship_type": 2,
        "verdict": "needs_human_call",
        "evidence": evidence,
        "reason": (
            f"Ambiguous zone: same sign, lift diff {diff:.2f}pp is between {TYPE2_NEAR_IDENTICAL_PP}pp and "
            f"{TYPE2_DISTINCT_SIGNAL_PP}pp, and no clear subset relationship -- not forcing a verdict."
        ),
    }


def _resolve_type3(pair: dict) -> dict:
    cat_col, num_col = (pair["feature_a"], pair["feature_b"]) if pair["type_a"] == "categorical" else (pair["feature_b"], pair["feature_a"])
    eta = pair["abs_value"]
    if eta > TYPE3_ETA_FLAG_THRESHOLD:
        return {
            "relationship_type": 3,
            "verdict": "needs_human_call",
            "categorical_column": cat_col,
            "continuous_column": num_col,
            "reason": (
                f"eta={eta} exceeds {TYPE3_ETA_FLAG_THRESHOLD} -- strong sign {cat_col} may just be a relabeling of "
                f"{num_col} rather than new information; this is a conceptual call this script cannot make."
            ),
        }
    return {
        "relationship_type": 3,
        "verdict": "keep_both",
        "categorical_column": cat_col,
        "continuous_column": num_col,
        "reason": f"eta={eta} <= {TYPE3_ETA_FLAG_THRESHOLD}; default is keep both (tactical/behavioural label is not just a relabeling of the raw measurement).",
    }


def resolve_dataset(dataset_key: str, correlation_data: dict) -> dict:
    dataset_cfg = DATASETS[dataset_key]
    correlation_ds = correlation_data["datasets"][dataset_key]
    review_pairs = correlation_ds["pairs"]["review"]
    dropped_columns = _dropped_columns_for_dataset(correlation_ds)

    df = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"])
    lifts = _load_lifts(df, dataset_cfg)

    resolved: list[dict] = []
    type1_boolean_verdict: dict[str, dict] = {}

    # Pass 1: resolve every pair that meets a structural/statistical test outright.
    deferred: list[dict] = []
    for pair in review_pairs:
        types = {pair["type_a"], pair["type_b"]}

        if _is_persistence_pair(pair["feature_a"], pair["feature_b"]):
            resolved.append({
                **pair,
                "relationship_type": 4,
                "verdict": "keep_both",
                "reason": "State-persistence / autocorrelation pair (same base concept vs its own lagged/prev-event version) -- not redundancy.",
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
            result = _resolve_type2(pair, df, lifts)
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
            # Continuous<->continuous (or continuous<->discrete) pairs aren't
            # covered by any of the four defined relationship types -- Type 1
            # is specifically boolean-vs-parent, Type 3 is categorical-vs-
            # continuous. Force-resolving these would mean guessing at a rule
            # the prompt never specified, so they are surfaced as
            # needs_human_call instead. Several of these (e.g. the passive
            # top_option_*/ball_x cluster) are exactly the pairs the
            # structural-redesign step downstream is meant to handle.
            resolved.append({
                **pair,
                "relationship_type": None,
                "verdict": "needs_human_call",
                "reason": (
                    "Continuous<->continuous pair; none of the four defined relationship types apply "
                    "(not boolean, not categorical, not a name-pattern persistence pair) -- no automated "
                    "rule covers this combination. May be a candidate for structural feature-engineering "
                    "review rather than a simple drop/keep call."
                ),
            })
            continue

        raise ValueError(f"Unhandled REVIEW pair type combination: {types} ({pair['feature_a']}/{pair['feature_b']})")

    # Pass 2: propagate an already-established boolean fate onto its
    # below-threshold Type-1 pairs, instead of flooding needs_human_call with
    # duplicate evidence of the same conclusion.
    for pair in deferred:
        bool_col, num_col = (pair["feature_a"], pair["feature_b"]) if pair["type_a"] == "boolean" else (pair["feature_b"], pair["feature_a"])
        prior = type1_boolean_verdict.get(bool_col)
        if prior is not None:
            resolved.append({
                **pair,
                "relationship_type": 1,
                "verdict": "superseded",
                "reason": (
                    f"{bool_col} is already resolved for removal by a separate, stronger pairing "
                    f"({prior['reason']}); this below-threshold pair (|r|={pair['abs_value']} < {TYPE1_R_THRESHOLD}) "
                    "is the same underlying redundancy, not independent evidence."
                ),
            })
        else:
            resolved.append({
                **pair,
                "relationship_type": None,
                "verdict": "needs_human_call",
                "reason": (
                    f"Boolean-continuous pair below the Type-1 structural threshold (|r|={pair['abs_value']} < "
                    f"{TYPE1_R_THRESHOLD}); no automated rule applies and {bool_col}'s fate isn't otherwise determined."
                ),
            })

    needs_human_call = [r for r in resolved if r["verdict"] == "needs_human_call"]

    # Informational only: categorical<->categorical/continuous "persistence"
    # and "conceptual relabeling" examples referenced by this pipeline's own
    # sanity table live in the COLLAPSE tier, not REVIEW, under this
    # codebase's tier thresholds (categorical pairs never reach REVIEW -- see
    # src/eda/correlation.py). Surfaced here for traceability, not re-litigated.
    collapse_categorical_context = [
        p for p in correlation_ds["pairs"]["collapse"]
        if {p["type_a"], p["type_b"]} & {"categorical"} and (
            _is_persistence_pair(p["feature_a"], p["feature_b"]) or p["method"] == "correlation_ratio"
        )
    ]

    return {
        "dataset": dataset_key,
        "n_review_pairs": len(review_pairs),
        "resolved_pairs": resolved,
        "needs_human_call": needs_human_call,
        "verdict_counts": pd.Series([r["verdict"] for r in resolved]).value_counts().to_dict(),
        "collapse_tier_categorical_context": {
            "note": (
                "Categorical<->categorical and categorical<->continuous pairs never land in the REVIEW tier "
                "under prompt 1/4's tier thresholds (categorical methods separate COLLAPSE from below-threshold "
                "directly, with no REVIEW band). These pairs are shown here for reference only; they were not "
                "processed by this script's Type 3/4 logic because they were never in its input."
            ),
            "pairs": collapse_categorical_context,
        },
    }


def main() -> None:
    correlation_data = json.loads(CORRELATION_PATH.read_text(encoding="utf-8"))

    results = {key: resolve_dataset(key, correlation_data) for key in DATASETS}

    output = {
        "generated_at": correlation_data["generated_at"],
        "source": str(CORRELATION_PATH.relative_to(REPO_ROOT)),
        "datasets": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n=== needs_human_call ===")
    for key, r in results.items():
        if not r["needs_human_call"]:
            print(f"\n{key}: none")
            continue
        print(f"\n{key}: {len(r['needs_human_call'])} pair(s)")
        for item in r["needs_human_call"]:
            print(f"  {item['feature_a']} <-> {item['feature_b']}  [{item.get('method')}, r={item.get('value')}]")
            print(f"    reason: {item['reason']}")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
