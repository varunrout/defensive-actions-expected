"""CLI entrypoint: numeric x numeric interaction analysis against
`target_xt_delta`, ACTIVE pairs only -- the xT sibling of
generate_feature_interaction_analysis_xg.py (the confirmed generator of the
xg portal's FEATURE_INTERACTION_ANALYSIS.json).

Carried over UNCHANGED:
  - the PAIRS list (imported), filtered to its ACTIVE entries; the passive
    pairs are out of scope for this prompt. Not re-ranked or re-selected --
    the whole point is that the same pairs are comparable across targets.
  - Q = 4 quartiles and `_bin_labeled`, imported.
  - SUBSTITUTIVE_RATIO_MAX = 0.3 and ADDITIVE_MAGNITUDE_RATIO_MAX = 2.0 --
    both are RATIOS of one delta to another, so they are scale-free and
    sign-free and need no adaptation. Same reasoning the xg version gives
    for reusing them from the binary version.
  - MIN_CELL_N_FOR_DELTA = 20.

ADAPTED: `min_marginal_delta`, the one threshold in this test that is an
absolute magnitude. The xg version sets it to 0.5 x the dataset's mean xg
(and, for its conditional panel, 0.5 x the shot-conditional mean xg).
Translated per Adaptation 1, using the unconditional and non-zero-delta
standard deviations respectively.

ADAPTED: the conditional panel is `target_xt_delta != 0` rather than
`target_future_shot_10s == 1` (Adaptation 2), so the JSON fields are named
`*_given_nonzero_delta` rather than `*_given_shot`.

Usage:
    python -m src.eda.generate_feature_interaction_analysis_xt
"""

from __future__ import annotations

import json

import pandas as pd

from src.eda import xt_common as xc
from src.eda.generate_feature_interaction_analysis import PAIRS, Q, _bin_labeled

OUTPUT_PATH = xc.OUT_DIR / "FEATURE_INTERACTION_ANALYSIS.json"
TARGET = xc.TARGET

SUBSTITUTIVE_RATIO_MAX = 0.3
ADDITIVE_MAGNITUDE_RATIO_MAX = 2.0
MIN_CELL_N_FOR_DELTA = 20


def _classify(marginal_delta: float | None, deltas: list[float], min_marginal_delta: float) -> tuple[str, str]:
    if marginal_delta is None or len(deltas) < 2:
        return "inconclusive", "Not enough populated quartile combinations to judge."
    if abs(marginal_delta) < min_marginal_delta:
        return "inconclusive", (
            f"Pooled marginal delta ({marginal_delta:+.6f}) is below the {min_marginal_delta:.6f} minimum to "
            "classify confidently. Note the |absolute value| test: on this target a large NEGATIVE marginal "
            "delta is just as classifiable as a large positive one."
        )

    abs_deltas = [abs(d) for d in deltas]
    mean_abs = sum(abs_deltas) / len(abs_deltas)
    ratio = mean_abs / abs(marginal_delta)

    if ratio < SUBSTITUTIVE_RATIO_MAX:
        return "substitutive", (
            f"Mean |within-B-stratum delta| ({mean_abs:.6f}) is {ratio:.2f}x the pooled marginal delta "
            f"({marginal_delta:+.6f}), below the {SUBSTITUTIVE_RATIO_MAX} threshold -- A's effect mostly "
            "disappears once B is known."
        )

    marginal_sign = marginal_delta >= 0
    sign_consistent = all((d >= 0) == marginal_sign for d in deltas)
    nonzero = [d for d in abs_deltas if d > 1e-12]
    magnitude_ratio = (max(nonzero) / min(nonzero)) if len(nonzero) >= 2 else 1.0

    if sign_consistent and magnitude_ratio <= ADDITIVE_MAGNITUDE_RATIO_MAX:
        return "additive", (
            f"Every B-stratum delta shares the marginal delta's sign and the max/min magnitude ratio "
            f"({magnitude_ratio:.2f}) is within the {ADDITIVE_MAGNITUDE_RATIO_MAX} threshold -- A's effect holds "
            "at roughly constant strength regardless of B's level."
        )

    detail = (
        "Sign flips across B-strata -- and on this target a sign flip means A's effect reverses DIRECTION "
        "(threat-reducing in one stratum, threat-increasing in another), not merely that it crosses a base rate"
        if not sign_consistent else
        f"Magnitude ratio ({magnitude_ratio:.2f}) exceeds {ADDITIVE_MAGNITUDE_RATIO_MAX}"
    )
    return "interactive", f"{detail} -- A's effect meaningfully depends on B's level."


def _delta_from_cells(cells, value_key, order_index):
    usable = sorted(
        (c for c in cells if c[value_key] is not None and c[_n_key(value_key)] >= MIN_CELL_N_FOR_DELTA),
        key=lambda c: order_index[c["a_bin"]],
    )
    if len(usable) < 2:
        return None, None, None
    return round(usable[-1][value_key] - usable[0][value_key], 8), usable[0], usable[-1]


def _n_key(value_key: str) -> str:
    return "n" if value_key == "mean_xt" else "n_given_nonzero_delta"


def analyze_pair(df: pd.DataFrame, spec: dict, min_delta: float, min_delta_nz: float) -> dict:
    fa, fb = spec["feature_a"], spec["feature_b"]
    sub = df[[fa, fb, TARGET]].dropna()

    a_bins = _bin_labeled(sub[fa])
    b_bins = _bin_labeled(sub[fb])
    a_labels, b_labels = list(a_bins.cat.categories), list(b_bins.cat.categories)
    order_index = {l: i for i, l in enumerate(a_labels)}

    nz_mask = sub[TARGET] != 0
    grouped = sub.groupby([b_bins, a_bins], observed=True)[TARGET].agg(n="count", mean_xt="mean")
    grouped_nz = sub.loc[nz_mask].groupby(
        [b_bins.loc[nz_mask], a_bins.loc[nz_mask]], observed=True)[TARGET].agg(n="count", mean_xt="mean")

    grid = []
    for b in b_labels:
        for a in a_labels:
            k = (b, a)
            n, m = (int(grouped.loc[k, "n"]), round(float(grouped.loc[k, "mean_xt"]), 8)) if k in grouped.index else (0, None)
            nnz, mnz = (int(grouped_nz.loc[k, "n"]), round(float(grouped_nz.loc[k, "mean_xt"]), 8)) if k in grouped_nz.index else (0, None)
            grid.append({"a_bin": a, "b_bin": b, "n": n, "mean_xt": m,
                         "n_given_nonzero_delta": nnz, "mean_xt_given_nonzero_delta": mnz})

    marginal = sub.groupby(a_bins, observed=True)[TARGET].agg(n="count", mean_xt="mean")
    marginal_nz = sub.loc[nz_mask].groupby(a_bins.loc[nz_mask], observed=True)[TARGET].agg(n="count", mean_xt="mean")
    m_cells = [{"a_bin": l, "n": int(marginal.loc[l, "n"]), "mean_xt": round(float(marginal.loc[l, "mean_xt"]), 8)}
               for l in a_labels]
    m_cells_nz = [
        {"a_bin": l, "n_given_nonzero_delta": int(marginal_nz.loc[l, "n"]),
         "mean_xt_given_nonzero_delta": round(float(marginal_nz.loc[l, "mean_xt"]), 8)}
        if l in marginal_nz.index else {"a_bin": l, "n_given_nonzero_delta": 0, "mean_xt_given_nonzero_delta": None}
        for l in a_labels
    ]
    m_delta, _, _ = _delta_from_cells(m_cells, "mean_xt", order_index)
    m_delta_nz, _, _ = _delta_from_cells(m_cells_nz, "mean_xt_given_nonzero_delta", order_index)

    grad, grad_nz, deltas, deltas_nz = [], [], [], []
    for b in b_labels:
        cells = [c for c in grid if c["b_bin"] == b]
        d, lo, hi = _delta_from_cells(cells, "mean_xt", order_index)
        grad.append({"b_bin": b, "a_delta_within_stratum": d,
                     "a_bin_compared_low": lo["a_bin"] if lo else None,
                     "a_bin_compared_high": hi["a_bin"] if hi else None,
                     "n_rows": sum(c["n"] for c in cells)})
        if d is not None:
            deltas.append(d)
        dz, loz, hiz = _delta_from_cells(cells, "mean_xt_given_nonzero_delta", order_index)
        grad_nz.append({"b_bin": b, "a_delta_within_stratum": dz,
                        "a_bin_compared_low": loz["a_bin"] if loz else None,
                        "a_bin_compared_high": hiz["a_bin"] if hiz else None,
                        "n_rows": sum(c["n_given_nonzero_delta"] for c in cells)})
        if dz is not None:
            deltas_nz.append(dz)

    cls, reason = _classify(m_delta, deltas, min_delta)
    cls_nz, reason_nz = _classify(m_delta_nz, deltas_nz, min_delta_nz)

    sign_flip = len(deltas) >= 2 and not all((d >= 0) == (deltas[0] >= 0) for d in deltas)

    return {
        "dataset": "active",
        "feature_a": fa, "feature_b": fb,
        "football_rationale": spec["football_rationale"],
        "a_labels": a_labels, "b_labels": b_labels,
        "n_rows_used": len(sub),
        "n_rows_used_given_nonzero_delta": int(nz_mask.sum()),
        "grid": grid,
        "marginal_a_by_quartile": m_cells,
        "marginal_a_by_quartile_given_nonzero_delta": m_cells_nz,
        "marginal_a_delta": m_delta,
        "marginal_a_delta_given_nonzero_delta": m_delta_nz,
        "a_gradient_by_b_stratum": grad,
        "a_gradient_by_b_stratum_given_nonzero_delta": grad_nz,
        "within_stratum_deltas_change_sign": sign_flip,
        "classification": cls,
        "classification_reason": reason,
        "classification_given_nonzero_delta": cls_nz,
        "classification_reason_given_nonzero_delta": reason_nz,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt()
    scale = xc.target_scale(df)

    active_pairs = [p for p in PAIRS if p["dataset"] == "active"]
    min_delta = scale["flat_margin"]
    min_delta_nz = scale["flat_margin_nonzero"]

    results = []
    for spec in active_pairs:
        r = analyze_pair(df, spec, min_delta, min_delta_nz)
        results.append(r)
        print(f"[active] {r['feature_a']} x {r['feature_b']}: unconditional={r['classification']}, "
              f"non-zero-delta={r['classification_given_nonzero_delta']}, sign-flip={r['within_stratum_deltas_change_sign']}")

    order = {"interactive": 0, "substitutive": 1, "additive": 2, "inconclusive": 3}
    ordered = sorted(results, key=lambda r: order[r["classification"]])
    n_agree = sum(1 for r in results if r["classification"] == r["classification_given_nonzero_delta"])

    output = {
        "dataset": "active",
        "leg": "active-binary only (passive leg out of scope for this target)",
        "target": TARGET,
        "target_scale": scale,
        "quartile_n": Q,
        "step_0_scope_note": (
            "The xg portal's FEATURE_INTERACTION_ANALYSIS covers all 10 pairs in "
            "generate_feature_interaction_analysis.PAIRS (5 active + 5 passive). This prompt is scoped to the "
            f"active leg, so the same imported list is filtered to its {len(active_pairs)} ACTIVE pairs -- a "
            "scope reduction, not a re-selection. The pairs are NOT re-ranked for this target: keeping the same "
            "pairs is what makes the three targets' results comparable."
        ),
        "thresholds": {
            "substitutive_ratio_max": SUBSTITUTIVE_RATIO_MAX,
            "additive_magnitude_ratio_max": ADDITIVE_MAGNITUDE_RATIO_MAX,
            "min_cell_n_for_delta": MIN_CELL_N_FOR_DELTA,
            "min_marginal_delta": round(min_delta, 8),
            "min_marginal_delta_given_nonzero_delta": round(min_delta_nz, 8),
            "note": (
                "The substitutive / additive classification cutoffs are RATIOS of one delta to another, so they "
                "are scale-free AND sign-free and are carried over completely unchanged -- the same reasoning "
                "the xg version gives for reusing them from the binary version. The one absolute-magnitude "
                "threshold, min_marginal_delta, is adapted: the xg version sets it to 0.5 x the dataset's own "
                "mean xg, which is meaningless for a target whose mean is -0.0013. It is translated instead via "
                "xg's own standard-deviation fraction, applied to target_xt_delta's standard deviation "
                "(unconditional) and to the non-zero-delta subset's standard deviation (conditional panel). "
                "The classification test uses |marginal delta|, so a large NEGATIVE marginal delta is as "
                "classifiable as a large positive one."
            ),
        },
        "candidate_selection": (
            "Same ACTIVE pairs as reports/analysis/shot_target/FEATURE_INTERACTION_ANALYSIS.json and the xg "
            "mirror -- see generate_feature_interaction_analysis.py's module docstring for the full ranking "
            "derivation and football rationale per pair. Computed here against target_xt_delta for direct "
            "comparability, with a non-zero-delta panel in place of the xg version's shot-conditional one."
        ),
        "n_pairs_total": len(results),
        "n_pairs_where_unconditional_and_conditional_agree": n_agree,
        "n_pairs_with_sign_flipping_stratum_deltas": sum(1 for r in results if r["within_stratum_deltas_change_sign"]),
        "summary_table": [
            {"dataset": r["dataset"], "feature_a": r["feature_a"], "feature_b": r["feature_b"],
             "classification": r["classification"],
             "classification_given_nonzero_delta": r["classification_given_nonzero_delta"],
             "within_stratum_deltas_change_sign": r["within_stratum_deltas_change_sign"]}
            for r in ordered
        ],
        "prompt_64_cross_references": {"distribution_shape": xc.PROMPT_64_SHAPE_FINDING},
        "pairs": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\n{n_agree}/{len(results)} pairs classify the same way unconditionally and on the non-zero-delta subset.")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
