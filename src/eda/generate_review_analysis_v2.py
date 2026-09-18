"""V2 (prompt 7/7) CLI entrypoint: resolve every REVIEW-tier pair from
CORRELATION_ANALYSIS_V2.json into a final verdict. Correlation alone never
decides a verdict here -- it only got the pair onto the REVIEW list. This
script uses actual shot-rate lift (Type 2), a structural boolean-vs-parent
test (Type 1), a conceptual categorical-context check (Type 5), an
engineered-family check (Type 6), or a naming-pattern autocorrelation check
(Type 4) as evidence.

This is the V2 companion to generate_review_analysis.py (V1, Types 1-4 only,
reads CORRELATION_ANALYSIS.json, writes REVIEW_ANALYSIS.json -- untouched by
this script). V2 additions: correlation_v2.py gives categorical methods
(Cramer's V, correlation ratio eta) their own REVIEW band, and this script's
Type 5 also reaches into the COLLAPSE tier for categorical<->continuous/
boolean pairs that were never processed by anything before (they used to go
straight from COLLAPSE to nowhere). Type 6 replaces the old blanket
"continuous<->continuous has no rule" fallback with an explicit, hardcoded
engineered-family check (feature_config_v2.FAMILY_GROUPS) -- family members
default to keep_both, everything else still honestly falls to
needs_human_call rather than forcing a family match that isn't real.

Pairs that genuinely can't be resolved by these tests are surfaced as
`needs_human_call: true`, never force-resolved.

Usage:
    python -m src.eda.generate_review_analysis_v2
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda import compute_stats as cs
from src.eda.feature_config_v2_historical import DATASETS_V2_HISTORICAL as DATASETS
from src.eda.feature_config_v2 import FAMILY_GROUPS, PART_B_RESOLVED_PAIRS

REPO_ROOT = Path(__file__).resolve().parents[2]
CORRELATION_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CORRELATION_ANALYSIS_V2.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "REVIEW_ANALYSIS_V2.json"

TYPE1_R_THRESHOLD = 0.75
TYPE2_R_THRESHOLD = 0.5
TYPE2_NEAR_IDENTICAL_PP = 1.0
TYPE2_DISTINCT_SIGNAL_PP = 3.0
TYPE2_SUBSET_CONTAINMENT = 0.90
TYPE5_ETA_FLAG_THRESHOLD = 0.85
TYPE5_V_FLAG_THRESHOLD = 0.7
TYPE6_R_FLAG_THRESHOLD = 0.90

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


def _resolve_type5(pair: dict) -> dict:
    """Categorical vs. its correlated continuous/boolean context.

    Conceptual test, not statistical: does the categorical column carry
    information beyond what's already in the continuous/boolean measurement,
    or is it just a relabeling of the same thing? Covers pairs from both the
    REVIEW tier (new categorical REVIEW band, prompt 7/7) and the COLLAPSE
    tier (categorical<->continuous/boolean pairs below the relabeling
    threshold that were never processed by anything before this).
    """
    key = frozenset({pair["feature_a"], pair["feature_b"]})
    override = PART_B_RESOLVED_PAIRS.get(key)
    if override is not None:
        verdict, reason = override
        return {"relationship_type": 5, "verdict": verdict, "reason": f"[Part B, checked against real data] {reason}"}

    cat_col, other_col = (pair["feature_a"], pair["feature_b"]) if pair["type_a"] == "categorical" else (pair["feature_b"], pair["feature_a"])
    magnitude = pair["abs_value"]
    threshold = TYPE5_ETA_FLAG_THRESHOLD if pair["method"] == "correlation_ratio" else TYPE5_V_FLAG_THRESHOLD
    stat_name = "eta" if pair["method"] == "correlation_ratio" else "Cramer's V"

    if magnitude >= threshold:
        return {
            "relationship_type": 5,
            "verdict": "needs_human_call",
            "categorical_column": cat_col,
            "other_column": other_col,
            "reason": (
                f"{stat_name}={magnitude} >= {threshold} -- strong sign {cat_col} may just be a relabeling of "
                f"{other_col} rather than new information; this is a conceptual call this script cannot make."
            ),
        }
    return {
        "relationship_type": 5,
        "verdict": "keep_both",
        "categorical_column": cat_col,
        "other_column": other_col,
        "reason": (
            f"{stat_name}={magnitude} < {threshold} -- tactical/contextual categorical correlating with geometry "
            f"or flag state, real football co-variation, not relabeling."
        ),
    }


def _find_family(dataset_key: str, col_a: str, col_b: str) -> dict | None:
    for group in FAMILY_GROUPS.get(dataset_key, []):
        if col_a in group["columns"] and col_b in group["columns"]:
            return group
    return None


def _resolve_type6(pair: dict, dataset_key: str) -> dict:
    """Continuous vs. continuous, same engineered family (e.g. *_within_5m
    vs *_within_10m, or ranked-option columns at different ranks).

    Family membership is an explicit, hardcoded list (feature_config.
    FAMILY_GROUPS), not inferred by string-matching -- a pair only
    auto-resolves when it's a genuine same-measurement-at-a-different-
    radius/rank relationship. Everything else honestly needs_human_call
    rather than forcing a family match that isn't real.
    """
    col_a, col_b = pair["feature_a"], pair["feature_b"]
    family = _find_family(dataset_key, col_a, col_b)

    if family is None:
        return {
            "relationship_type": 6,
            "verdict": "needs_human_call",
            "reason": (
                "Continuous<->continuous pair with no matching entry in feature_config.FAMILY_GROUPS -- not a "
                "recognized same-measurement-at-different-radius/rank/option relationship, so no automated rule "
                "applies. May be a candidate for structural feature-engineering review rather than a simple "
                "drop/keep call."
            ),
        }

    if family.get("always_needs_human_call"):
        return {
            "relationship_type": 6,
            "verdict": "needs_human_call",
            "reason": family["always_reason"],
        }

    if pair["abs_value"] >= TYPE6_R_FLAG_THRESHOLD:
        return {
            "relationship_type": 6,
            "verdict": "needs_human_call",
            "reason": (
                f"Same engineered family ({family['reason_tag']}) but |r|={pair['abs_value']} >= "
                f"{TYPE6_R_FLAG_THRESHOLD} -- close enough to warrant a look rather than an automatic keep-both "
                "(same bound used for the V1 COLLAPSE clusters)."
            ),
        }

    return {
        "relationship_type": 6,
        "verdict": "keep_both",
        "reason": (
            f"Same engineered family ({family['reason_tag']}) -- expected to correlate, not redundant; deferred "
            "to feature-importance evidence from a baseline model, not resolved by correlation alone."
        ),
    }


def resolve_dataset(dataset_key: str, correlation_data: dict) -> dict:
    dataset_cfg = DATASETS[dataset_key]
    correlation_ds = correlation_data["datasets"][dataset_key]
    review_pairs = correlation_ds["pairs"]["review"]
    dropped_columns = _dropped_columns_for_dataset(correlation_ds)

    df = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"])
    lifts = _load_lifts(df, dataset_cfg)

    # V2 (prompt 7/7): Type 5 also reaches into COLLAPSE for
    # categorical<->continuous/boolean pairs -- those never had any resolver
    # applied to them before (COLLAPSE-tier categorical pairs were tiered and
    # then never touched by generate_review_analysis.py at all).
    collapse_categorical_pairs = [
        p for p in correlation_ds["pairs"]["collapse"]
        if p["method"] in ("cramers_v", "correlation_ratio")
    ]
    # Same symmetry for Type 6: a numeric<->numeric pair that happens to land
    # in COLLAPSE (e.g. top_option_2_threat_score<->top_option_3_threat_score
    # at r=0.9024, just over the 0.90 COLLAPSE cut) but matches a known
    # engineered family still needs the family's verdict applied -- otherwise
    # a Cluster-5 pair silently escapes needs_human_call just because it
    # crossed a tier boundary this script doesn't otherwise look at.
    collapse_family_numeric_pairs = [
        p for p in correlation_ds["pairs"]["collapse"]
        if p["type_a"] == "numeric" and p["type_b"] == "numeric"
        and _find_family(dataset_key, p["feature_a"], p["feature_b"]) is not None
    ]
    n_collapse_categorical = len(collapse_categorical_pairs) + len(collapse_family_numeric_pairs)

    resolved: list[dict] = []
    type1_boolean_verdict: dict[str, dict] = {}

    tagged_pairs = (
        [(p, "review") for p in review_pairs]
        + [(p, "collapse") for p in collapse_categorical_pairs]
        + [(p, "collapse") for p in collapse_family_numeric_pairs]
    )

    # Pass 1: resolve every pair that meets a structural/statistical test outright.
    deferred: list[tuple[dict, str]] = []
    for pair, source_tier in tagged_pairs:
        types = {pair["type_a"], pair["type_b"]}

        if _is_persistence_pair(pair["feature_a"], pair["feature_b"]):
            resolved.append({
                **pair,
                "source_tier": source_tier,
                "relationship_type": 4,
                "verdict": "keep_both",
                "reason": "State-persistence / autocorrelation pair (same base concept vs its own lagged/prev-event version) -- not redundancy.",
            })
            continue

        if types == {"boolean", "numeric"}:
            if pair["abs_value"] >= TYPE1_R_THRESHOLD:
                result = _resolve_type1(pair, dropped_columns)
                resolved.append({**pair, "source_tier": source_tier, **result})
                if result["verdict"] in ("drop_boolean", "moot_after_drop"):
                    type1_boolean_verdict[result["boolean_column"]] = result
            else:
                deferred.append((pair, source_tier))
            continue

        if types == {"boolean"}:
            result = _resolve_type2(pair, df, lifts)
            resolved.append({**pair, "source_tier": source_tier, **result})
            if result["verdict"].startswith("drop_"):
                drop_col = pair["feature_a"] if result["verdict"] == "drop_a" else pair["feature_b"]
                type1_boolean_verdict.setdefault(drop_col, {"verdict": "drop_boolean", "reason": result["reason"]})
            continue

        if "categorical" in types:
            result = _resolve_type5(pair)
            resolved.append({**pair, "source_tier": source_tier, **result})
            continue

        if types == {"numeric"}:
            result = _resolve_type6(pair, dataset_key)
            resolved.append({**pair, "source_tier": source_tier, **result})
            continue

        raise ValueError(f"Unhandled pair type combination: {types} ({pair['feature_a']}/{pair['feature_b']})")

    # Pass 2: propagate an already-established boolean fate onto its
    # below-threshold Type-1 pairs, instead of flooding needs_human_call with
    # duplicate evidence of the same conclusion.
    for pair, source_tier in deferred:
        bool_col, num_col = (pair["feature_a"], pair["feature_b"]) if pair["type_a"] == "boolean" else (pair["feature_b"], pair["feature_a"])
        prior = type1_boolean_verdict.get(bool_col)
        if prior is not None:
            resolved.append({
                **pair,
                "source_tier": source_tier,
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
                "source_tier": source_tier,
                "relationship_type": None,
                "verdict": "needs_human_call",
                "reason": (
                    f"Boolean-continuous pair below the Type-1 structural threshold (|r|={pair['abs_value']} < "
                    f"{TYPE1_R_THRESHOLD}); no automated rule applies and {bool_col}'s fate isn't otherwise determined."
                ),
            })

    needs_human_call = [r for r in resolved if r["verdict"] == "needs_human_call"]

    return {
        "dataset": dataset_key,
        "n_review_pairs": len(review_pairs),
        "n_collapse_categorical_pairs_processed": n_collapse_categorical,
        "resolved_pairs": resolved,
        "needs_human_call": needs_human_call,
        "verdict_counts": pd.Series([r["verdict"] for r in resolved]).value_counts().to_dict(),
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
