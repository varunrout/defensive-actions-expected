"""CLI entrypoint: prompt 30 -- numeric x numeric interaction analysis (xG /
continuous target). Same 10 pairs, same candidate-selection reasoning, same
4x4 quartile-grid method as generate_feature_interaction_analysis.py
(binary), against mean target_future_xg_10s instead of shot-rate
percentage, with a shot-conditional (target_future_shot_10s == 1) panel
computed alongside the unconditional one from the start -- same reasoning
prompt 23/26 established: xg is structurally zero wherever no shot occurs,
so the unconditional mean_xg mostly re-derives occurrence; the
shot-conditional numbers are where genuine chance-quality signal shows up.

Both panels get their own additive/interactive/substitutive classification
(classification / classification_given_shot) since the two can legitimately
disagree, same as prompt 26's occurrence_vs_quality distinction.

flat-margin-style thresholds (min_marginal_delta) are relative to each
dataset's own overall mean xG (unconditional) or shot-conditional mean xG
(given-shot) -- same FLAT_MARGIN_RATIO convention as
generate_numerical_xg_target_analysis.py, not a fixed value like the
binary version's 1.0pp (xG's scale is far smaller and not comparable to
percentage points).

Usage:
    python -m src.eda.generate_feature_interaction_analysis_xg
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_feature_interaction_analysis import PAIRS, Q, _bin_labeled
from src.eda.generate_numerical_xg_target_analysis import FLAT_MARGIN_RATIO, SHOT_COL, TARGET

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "xg_target" / "FEATURE_INTERACTION_ANALYSIS.json"

SUBSTITUTIVE_RATIO_MAX = 0.3
ADDITIVE_MAGNITUDE_RATIO_MAX = 2.0
MIN_CELL_N_FOR_DELTA = 20


def _classify(marginal_delta: float | None, deltas: list[float], min_marginal_delta: float) -> tuple[str, str]:
    if marginal_delta is None or len(deltas) < 2:
        return "inconclusive", "Not enough populated quartile combinations to judge."
    if abs(marginal_delta) < min_marginal_delta:
        return "inconclusive", f"Pooled marginal delta ({marginal_delta:+.6f}) is below the {min_marginal_delta:.6f} minimum to classify confidently."

    abs_deltas = [abs(d) for d in deltas]
    mean_abs_delta = sum(abs_deltas) / len(abs_deltas)
    ratio_to_marginal = mean_abs_delta / abs(marginal_delta)

    if ratio_to_marginal < SUBSTITUTIVE_RATIO_MAX:
        return "substitutive", (
            f"Mean |within-B-stratum delta| ({mean_abs_delta:.6f}) is {ratio_to_marginal:.2f}x the pooled marginal "
            f"delta ({marginal_delta:+.6f}), below the {SUBSTITUTIVE_RATIO_MAX} threshold -- A's effect mostly "
            "disappears once B is known."
        )

    marginal_sign = marginal_delta >= 0
    sign_consistent = all((d >= 0) == marginal_sign for d in deltas)
    nonzero_abs = [d for d in abs_deltas if d > 1e-12]
    magnitude_ratio = (max(nonzero_abs) / min(nonzero_abs)) if len(nonzero_abs) >= 2 else 1.0

    if sign_consistent and magnitude_ratio <= ADDITIVE_MAGNITUDE_RATIO_MAX:
        return "additive", (
            f"Every B-stratum delta shares the marginal delta's sign and the max/min magnitude ratio "
            f"({magnitude_ratio:.2f}) is within the {ADDITIVE_MAGNITUDE_RATIO_MAX} threshold -- A's effect holds "
            "at roughly constant strength regardless of B's level."
        )

    return "interactive", (
        f"{'Sign flips across B-strata' if not sign_consistent else f'Magnitude ratio ({magnitude_ratio:.2f}) exceeds {ADDITIVE_MAGNITUDE_RATIO_MAX}'} "
        "-- A's effect meaningfully depends on B's level."
    )


def _delta_from_cells(cells: list[dict], value_key: str, order_index: dict) -> tuple[float | None, dict | None, dict | None]:
    usable = sorted(
        (c for c in cells if c[value_key] is not None and c["n"] >= MIN_CELL_N_FOR_DELTA),
        key=lambda c: order_index[c["a_bin"]],
    )
    if len(usable) < 2:
        return None, None, None
    return round(usable[-1][value_key] - usable[0][value_key], 6), usable[0], usable[-1]


def analyze_pair(df: pd.DataFrame, spec: dict, min_marginal_delta: float, min_marginal_delta_given_shot: float) -> dict:
    feature_a, feature_b = spec["feature_a"], spec["feature_b"]
    sub = df[[feature_a, feature_b, TARGET, SHOT_COL]].dropna(subset=[feature_a, feature_b, TARGET])

    a_bins = _bin_labeled(sub[feature_a])
    b_bins = _bin_labeled(sub[feature_b])
    a_labels = list(a_bins.cat.categories)
    b_labels = list(b_bins.cat.categories)
    order_index = {lbl: i for i, lbl in enumerate(a_labels)}

    shot_mask = sub[SHOT_COL] == 1
    grouped = sub.groupby([b_bins, a_bins], observed=True)[TARGET].agg(n="count", mean_xg="mean")
    grouped_gs = sub.loc[shot_mask].groupby([b_bins.loc[shot_mask], a_bins.loc[shot_mask]], observed=True)[TARGET].agg(n="count", mean_xg="mean")

    grid = []
    for b_label in b_labels:
        for a_label in a_labels:
            key = (b_label, a_label)
            if key in grouped.index:
                r = grouped.loc[key]
                n, mean_xg = int(r["n"]), round(float(r["mean_xg"]), 6)
            else:
                n, mean_xg = 0, None
            if key in grouped_gs.index:
                rg = grouped_gs.loc[key]
                n_gs, mean_xg_gs = int(rg["n"]), round(float(rg["mean_xg"]), 6)
            else:
                n_gs, mean_xg_gs = 0, None
            grid.append({
                "a_bin": a_label, "b_bin": b_label, "n": n, "mean_xg": mean_xg,
                "n_given_shot": n_gs, "mean_xg_given_shot": mean_xg_gs,
            })

    marginal = sub.groupby(a_bins, observed=True)[TARGET].agg(n="count", mean_xg="mean")
    marginal_gs = sub.loc[shot_mask].groupby(a_bins.loc[shot_mask], observed=True)[TARGET].agg(n="count", mean_xg="mean")
    marginal_cells = [
        {"a_bin": lbl, "n": int(marginal.loc[lbl, "n"]), "mean_xg": round(float(marginal.loc[lbl, "mean_xg"]), 6)}
        for lbl in a_labels
    ]
    marginal_cells_gs = [
        {"a_bin": lbl, "n": int(marginal_gs.loc[lbl, "n"]), "mean_xg": round(float(marginal_gs.loc[lbl, "mean_xg"]), 6)}
        if lbl in marginal_gs.index else {"a_bin": lbl, "n": 0, "mean_xg": None}
        for lbl in a_labels
    ]
    marginal_delta, _, _ = _delta_from_cells(marginal_cells, "mean_xg", order_index)
    marginal_delta_gs, _, _ = _delta_from_cells(marginal_cells_gs, "mean_xg", order_index)

    gradient_by_b, gradient_by_b_gs = [], []
    deltas, deltas_gs = [], []
    for b_label in b_labels:
        cells_in_stratum = [c for c in grid if c["b_bin"] == b_label]
        delta, lo, hi = _delta_from_cells(cells_in_stratum, "mean_xg", order_index)
        n_in_stratum = sum(c["n"] for c in cells_in_stratum)
        gradient_by_b.append({
            "b_bin": b_label, "a_delta_within_stratum": delta,
            "a_bin_compared_low": lo["a_bin"] if lo else None, "a_bin_compared_high": hi["a_bin"] if hi else None,
            "n_rows": n_in_stratum,
        })
        if delta is not None:
            deltas.append(delta)

        delta_gs, lo_gs, hi_gs = _delta_from_cells(cells_in_stratum, "mean_xg_given_shot", order_index)
        n_gs_in_stratum = sum(c["n_given_shot"] for c in cells_in_stratum)
        gradient_by_b_gs.append({
            "b_bin": b_label, "a_delta_within_stratum": delta_gs,
            "a_bin_compared_low": lo_gs["a_bin"] if lo_gs else None, "a_bin_compared_high": hi_gs["a_bin"] if hi_gs else None,
            "n_rows": n_gs_in_stratum,
        })
        if delta_gs is not None:
            deltas_gs.append(delta_gs)

    classification, reason = _classify(marginal_delta, deltas, min_marginal_delta)
    classification_gs, reason_gs = _classify(marginal_delta_gs, deltas_gs, min_marginal_delta_given_shot)

    return {
        "dataset": spec["dataset"],
        "feature_a": feature_a,
        "feature_b": feature_b,
        "football_rationale": spec["football_rationale"],
        "a_labels": a_labels,
        "b_labels": b_labels,
        "n_rows_used": len(sub),
        "n_rows_used_given_shot": int(shot_mask.sum()),
        "grid": grid,
        "marginal_a_by_quartile": marginal_cells,
        "marginal_a_by_quartile_given_shot": marginal_cells_gs,
        "marginal_a_delta": marginal_delta,
        "marginal_a_delta_given_shot": marginal_delta_gs,
        "a_gradient_by_b_stratum": gradient_by_b,
        "a_gradient_by_b_stratum_given_shot": gradient_by_b_gs,
        "classification": classification,
        "classification_reason": reason,
        "classification_given_shot": classification_gs,
        "classification_reason_given_shot": reason_gs,
    }


def main() -> None:
    active_df = pd.read_parquet(REPO_ROOT / ACTIVE["parquet_path"])
    passive_df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])
    dfs = {"active": active_df, "passive": passive_df}

    min_marginal_deltas = {}
    min_marginal_deltas_gs = {}
    for ds, df in dfs.items():
        overall_mean = float(df[TARGET].mean())
        shot_mean = float(df.loc[df[SHOT_COL] == 1, TARGET].mean())
        min_marginal_deltas[ds] = FLAT_MARGIN_RATIO * overall_mean
        min_marginal_deltas_gs[ds] = FLAT_MARGIN_RATIO * shot_mean

    results = []
    for spec in PAIRS:
        ds = spec["dataset"]
        result = analyze_pair(dfs[ds], spec, min_marginal_deltas[ds], min_marginal_deltas_gs[ds])
        results.append(result)
        print(f"[{ds}] {result['feature_a']} x {result['feature_b']}: "
              f"unconditional={result['classification']}, given-shot={result['classification_given_shot']}")

    order = {"interactive": 0, "substitutive": 1, "additive": 2, "inconclusive": 3}
    results_sorted = sorted(results, key=lambda r: order[r["classification"]])
    summary_table = [
        {
            "dataset": r["dataset"], "feature_a": r["feature_a"], "feature_b": r["feature_b"],
            "classification": r["classification"], "classification_given_shot": r["classification_given_shot"],
        }
        for r in results_sorted
    ]

    n_agree = sum(1 for r in results if r["classification"] == r["classification_given_shot"])

    output = {
        "target": TARGET,
        "shot_col": SHOT_COL,
        "quartile_n": Q,
        "thresholds": {
            "substitutive_ratio_max": SUBSTITUTIVE_RATIO_MAX,
            "additive_magnitude_ratio_max": ADDITIVE_MAGNITUDE_RATIO_MAX,
            "min_cell_n_for_delta": MIN_CELL_N_FOR_DELTA,
            "flat_margin_ratio": FLAT_MARGIN_RATIO,
            "min_marginal_delta_by_dataset": {k: round(v, 6) for k, v in min_marginal_deltas.items()},
            "min_marginal_delta_given_shot_by_dataset": {k: round(v, 6) for k, v in min_marginal_deltas_gs.items()},
            "note": (
                "Same classification logic as the binary-target file (substitutive / additive / interactive "
                "ratio thresholds are scale-free and identical), but the minimum-marginal-delta-to-classify "
                "threshold is relative to each dataset's own overall mean xG (unconditional) or shot-conditional "
                "mean xG (given-shot) -- flat_margin_ratio x that mean, same convention as "
                "generate_numerical_xg_target_analysis.py -- rather than a fixed percentage-point value, since "
                "xG's scale (~0.008 unconditional, ~10x higher given-shot) isn't comparable to a shot-rate "
                "percentage."
            ),
        },
        "candidate_selection": (
            "Same 10 pairs as reports/analysis/shot_target/FEATURE_INTERACTION_ANALYSIS.json (binary target) -- see that file's "
            "candidate_selection field and this repo's generate_feature_interaction_analysis.py module docstring "
            "for the full ranking derivation and football rationale per pair. Computed here against "
            "target_future_xg_10s for direct comparability, with a shot-conditional panel alongside the "
            "unconditional one from the start."
        ),
        "n_pairs_where_unconditional_and_given_shot_agree": n_agree,
        "n_pairs_total": len(results),
        "summary_table": summary_table,
        "pairs": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\n{n_agree}/{len(results)} pairs classify the same way unconditionally and given-shot.")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
