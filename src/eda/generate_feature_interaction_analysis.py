"""CLI entrypoint: prompt 30 -- numeric x numeric interaction analysis
(binary target). Everything tested so far pairs one numerical feature
against one categorical slicer (prompts 24-28), or tests a numeric-vs-numeric
CONFOUND with 1D quartile stratification (existing CONFOUND_ANALYSIS.json
tests). This is the first 2D joint analysis: two continuous/discrete locked
features binned into quartiles together (4x4 grid), checking whether
feature A's marginal gradient is stable across feature B's levels.

Step 1 -- candidate selection, from evidence already in hand:
Read reports/eda/SLICE_STRATIFICATION.json's cross_dataset_summary.
genuine_divergences (320 entries) and collapse each dataset's slicers to
prompt 27's CONSENSUS redundancy clusters (both metrics agree) before
counting how many independent slicers each feature diverges on -- active
collapses phase_label_prev_event into phase_label (the only consensus
cluster prompt 27 found); passive has no consensus clusters, all 4 slicers
stay separate. position_group (used as a slicer only in prompt 24's
original cells, not a locked feature) is treated as the same information as
`position` for this collapse -- feature_config.py confirms Cramer's V=1.0
between them (REDUNDANCY_DROPPED_ACTIVE), so counting both would double-count
the same evidence exactly the way prompt 27 warned about. Ties broken by
total raw divergence-row count (more unstable overall), then alphabetically.

Top 5 active: defenders_within_10m, visible_defender_count,
defenders_within_5m, possession_elapsed_seconds,
defenders_between_ball_and_attacking_goal.
Top 5 passive: top_option_2_threat_score, lane_screening_score_option_2,
marking_tightness, overload_score, lane_screening_score_option_1.

Step 2 -- partner selection: one other locked numeric feature per top-5
feature, chosen on football-domain grounds (stated per pair below), never
a pair already tested identically in CONFOUND_ANALYSIS.json's 1D tests.

Method: both features quartiled (pd.qcut, q=4, duplicates='drop' -- a few
low-cardinality discrete features collapse to fewer than 4 actual bins,
reported honestly rather than forced). For each of B's quartiles, compute
A's first-to-last-quartile shot-rate delta (the same "does the gradient
survive" logic as the existing 1D confound tests, just repeated once per
B-stratum instead of once overall). Classification (thresholds stated
explicitly, not left implicit):
  - substitutive: mean |within-B-stratum delta| < 0.3x |A's pooled marginal
    delta| -- A's effect is mostly gone once B is known.
  - additive: not substitutive, every stratum's delta shares the marginal
    delta's sign, and max/min |delta| ratio across strata <= 2.0 -- A's
    effect holds at roughly constant strength regardless of B.
  - interactive: anything else (sign flips across strata, or magnitude
    ratio > 2.0) -- A's effect meaningfully depends on B's level.
A |marginal delta| below 1.0pp is treated as too small to classify
confidently either way and flagged inconclusive.

Usage:
    python -m src.eda.generate_feature_interaction_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import ACTIVE, PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "FEATURE_INTERACTION_ANALYSIS.json"

TARGET = "target_future_shot_10s"
Q = 4
SUBSTITUTIVE_RATIO_MAX = 0.3
ADDITIVE_MAGNITUDE_RATIO_MAX = 2.0
MIN_MARGINAL_DELTA_PP = 1.0
MIN_CELL_N_FOR_DELTA = 20

PAIRS = [
    {
        "dataset": "active", "feature_a": "defenders_within_10m", "feature_b": "distance_to_attacking_box",
        "football_rationale": (
            "Defenders converge near the ball more as play nears their own box -- pitch geometry compresses "
            "everyone into a smaller space close to goal. Is 'defenders within 10m' just tracking how deep in "
            "the defensive third the action is, rather than a deliberate compactness/pressing choice?"
        ),
    },
    {
        "dataset": "active", "feature_a": "visible_defender_count", "feature_b": "attacker_spread",
        "football_rationale": (
            "A wider attacking shape pulls more defenders across the width of the pitch and into the tracked "
            "frame. Does the raw count of defenders on screen just mirror how spread out the attack already is, "
            "rather than reflecting a deliberate defensive commitment of numbers?"
        ),
    },
    {
        "dataset": "active", "feature_a": "defenders_within_5m", "feature_b": "defenders_within_10m",
        "football_rationale": (
            "Nested by construction -- every defender within 5m is also within 10m, so the two counts are not "
            "independent measurements. Once the 10m count is known, does the tighter 5m ring add real extra "
            "information about pressing intensity, or is it substitutively redundant with the wider count?"
        ),
    },
    {
        "dataset": "active", "feature_a": "possession_elapsed_seconds", "feature_b": "match_time_seconds",
        "football_rationale": (
            "Long-running possessions cluster in certain match phases (e.g. a team camped in the attacking third "
            "late in a half). Does the possession-clock effect just track match-clock context, rather than "
            "reflecting fatigue or space actually created by a sustained spell of pressure?"
        ),
    },
    {
        "dataset": "active", "feature_a": "defenders_between_ball_and_attacking_goal", "feature_b": "attacker_defender_ratio",
        "football_rationale": (
            "'Defenders goal-side of the ball' is a positioning-specific count, but it could just be restating "
            "the team's overall numerical balance on the pitch. Once the attacker:defender ratio is known, does "
            "goal-side positioning add anything beyond raw defensive numbers?"
        ),
    },
    {
        "dataset": "passive", "feature_a": "top_option_2_threat_score", "feature_b": "top_option_2_distance_from_ball",
        "football_rationale": (
            "The threat-score model for a passing option plausibly weights proximity to the ball heavily -- a "
            "closer option is easier to find and more immediately dangerous. Does option 2's threat score add "
            "positional-danger information beyond what its raw distance from the ball already says?"
        ),
    },
    {
        "dataset": "passive", "feature_a": "lane_screening_score_option_2", "feature_b": "engagement_distance_to_carrier",
        "football_rationale": (
            "A defender tightly engaging the ball-carrier is, almost by geometry, closer to more passing lanes. "
            "Is screening credit for option 2 mostly a byproduct of how tightly the defender is already engaging "
            "the carrier, rather than a deliberate lane-blocking position?"
        ),
    },
    {
        "dataset": "passive", "feature_a": "marking_tightness", "feature_b": "engagement_distance_to_carrier",
        "football_rationale": (
            "Both are distance measurements from a defender to a football-relevant point (nearest attacker vs "
            "the ball-carrier specifically) -- plausibly the same underlying 'how close is this defender to the "
            "action' phenomenon described twice, not two independent defensive behaviours."
        ),
    },
    {
        "dataset": "passive", "feature_a": "overload_score", "feature_b": "attacking_goal_centrality",
        "football_rationale": (
            "Central zones near goal draw more covering defenders by default (it's the most dangerous area to "
            "leave open). Does defensive overload just track how central the action is, rather than reflecting a "
            "genuine tactical over-commitment of numbers?"
        ),
    },
    {
        "dataset": "passive", "feature_a": "lane_screening_score_option_1", "feature_b": "marking_tightness",
        "football_rationale": (
            "Both come from the same defender-slot row and could both just be capturing 'this defender is "
            "positioned well' in general. Does lane-screening credit for the top option add independent "
            "blocking-position signal, or is it mostly redundant with how tightly the defender is already marking?"
        ),
    },
]


def _quartile_labels(cats: pd.Categorical) -> list[str]:
    n = len(cats.categories)
    return [f"Q{i + 1}" for i in range(n)]


def _bin_labeled(series: pd.Series) -> pd.Series:
    """Quartile by default (pd.qcut, q=4, duplicates='drop'). Falls back to
    exact-value grouping only when qcut can't produce at least 3 usable
    bins -- a heavily skewed low-cardinality discrete feature (overload_score
    is ~76% zeros, confirmed empirically: q=4 collapses to exactly 1 bin)
    would otherwise silently degenerate to 1-2 bins instead of a real
    quartile split. Most discrete features here (9-12 unique values,
    e.g. defenders_within_10m) qcut into a real 4-way split fine and don't
    need this fallback."""
    cats = pd.qcut(series, q=Q, duplicates="drop")
    if len(cats.cat.categories) >= 3:
        labels = _quartile_labels(cats.cat)
        return cats.cat.rename_categories(labels)
    return series.astype("category")


def analyze_pair(df: pd.DataFrame, spec: dict) -> dict:
    feature_a, feature_b = spec["feature_a"], spec["feature_b"]
    sub = df[[feature_a, feature_b, TARGET]].dropna()

    a_bins = _bin_labeled(sub[feature_a])
    b_bins = _bin_labeled(sub[feature_b])
    a_labels = list(a_bins.cat.categories)
    b_labels = list(b_bins.cat.categories)

    grouped = sub.groupby([b_bins, a_bins], observed=True)[TARGET].agg(n="count", shots="sum", rate=lambda s: s.mean() * 100)

    grid = []
    for b_label in b_labels:
        for a_label in a_labels:
            key = (b_label, a_label)
            if key in grouped.index:
                r = grouped.loc[key]
                n, shots, rate = int(r["n"]), int(r["shots"]), round(float(r["rate"]), 3)
            else:
                n, shots, rate = 0, 0, None
            grid.append({"a_bin": a_label, "b_bin": b_label, "n": n, "shots": shots, "rate": rate})

    # Pooled marginal A delta (ignoring B entirely).
    marginal_counts = sub.groupby(a_bins, observed=True)[TARGET].agg(n="count", rate=lambda s: s.mean() * 100)
    marginal_rates = [round(float(marginal_counts.loc[lbl, "rate"]), 3) for lbl in a_labels]
    marginal_usable_labels = [lbl for lbl in a_labels if marginal_counts.loc[lbl, "n"] >= MIN_CELL_N_FOR_DELTA]
    marginal_delta = (
        round(float(marginal_counts.loc[marginal_usable_labels[-1], "rate"]) - float(marginal_counts.loc[marginal_usable_labels[0], "rate"]), 3)
        if len(marginal_usable_labels) >= 2 else None
    )

    order_index = {lbl: i for i, lbl in enumerate(a_labels)}
    gradient_by_b = []
    deltas = []
    for b_label in b_labels:
        cells_in_stratum = [c for c in grid if c["b_bin"] == b_label]
        # Compare the lowest and highest A-bin that both have data AND at
        # least MIN_CELL_N_FOR_DELTA rows -- a sparse tail bin (e.g. n=2 in
        # a heavily skewed discrete feature) produces a delta driven by
        # near-noise, not a real gradient endpoint.
        usable = sorted(
            (c for c in cells_in_stratum if c["rate"] is not None and c["n"] >= MIN_CELL_N_FOR_DELTA),
            key=lambda c: order_index[c["a_bin"]],
        )
        delta = round(usable[-1]["rate"] - usable[0]["rate"], 3) if len(usable) >= 2 else None
        n_in_stratum = sum(c["n"] for c in cells_in_stratum)
        gradient_by_b.append({
            "b_bin": b_label,
            "a_delta_within_stratum": delta,
            "a_bin_compared_low": usable[0]["a_bin"] if usable else None,
            "a_bin_compared_high": usable[-1]["a_bin"] if usable else None,
            "n_populated_a_bins": len(usable),
            "n_rows": n_in_stratum,
        })
        if delta is not None:
            deltas.append(delta)

    classification, reason = _classify(marginal_delta, deltas)

    return {
        "dataset": spec["dataset"],
        "feature_a": feature_a,
        "feature_b": feature_b,
        "football_rationale": spec["football_rationale"],
        "a_labels": a_labels,
        "b_labels": b_labels,
        "n_rows_used": len(sub),
        "grid": grid,
        "marginal_a_rates_by_quartile": list(zip(a_labels, marginal_rates)),
        "marginal_a_delta_pp": marginal_delta,
        "a_gradient_by_b_stratum": gradient_by_b,
        "classification": classification,
        "classification_reason": reason,
    }


def _classify(marginal_delta: float | None, deltas: list[float]) -> tuple[str, str]:
    if marginal_delta is None or len(deltas) < 2:
        return "inconclusive", "Not enough populated quartile combinations to judge."
    if abs(marginal_delta) < MIN_MARGINAL_DELTA_PP:
        return "inconclusive", f"Pooled marginal delta ({marginal_delta:+.3f}pp) is below the {MIN_MARGINAL_DELTA_PP}pp minimum to classify confidently."

    abs_deltas = [abs(d) for d in deltas]
    mean_abs_delta = sum(abs_deltas) / len(abs_deltas)
    ratio_to_marginal = mean_abs_delta / abs(marginal_delta)

    if ratio_to_marginal < SUBSTITUTIVE_RATIO_MAX:
        return "substitutive", (
            f"Mean |within-B-stratum delta| ({mean_abs_delta:.3f}pp) is {ratio_to_marginal:.2f}x the pooled "
            f"marginal delta ({marginal_delta:+.3f}pp), below the {SUBSTITUTIVE_RATIO_MAX} threshold -- A's effect "
            "mostly disappears once B is known."
        )

    marginal_sign = marginal_delta >= 0
    sign_consistent = all((d >= 0) == marginal_sign for d in deltas)
    nonzero_abs = [d for d in abs_deltas if d > 1e-9]
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


def main() -> None:
    active_df = pd.read_parquet(REPO_ROOT / ACTIVE["parquet_path"])
    passive_df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])
    dfs = {"active": active_df, "passive": passive_df}

    results = []
    for spec in PAIRS:
        result = analyze_pair(dfs[spec["dataset"]], spec)
        results.append(result)
        print(f"[{result['dataset']}] {result['feature_a']} x {result['feature_b']}: {result['classification']} -- {result['classification_reason']}")

    results_sorted = sorted(results, key=lambda r: {"interactive": 0, "substitutive": 1, "additive": 2, "inconclusive": 3}[r["classification"]])
    summary_table = [
        {"dataset": r["dataset"], "feature_a": r["feature_a"], "feature_b": r["feature_b"], "classification": r["classification"]}
        for r in results_sorted
    ]

    output = {
        "target": TARGET,
        "quartile_n": Q,
        "thresholds": {
            "substitutive_ratio_max": SUBSTITUTIVE_RATIO_MAX,
            "additive_magnitude_ratio_max": ADDITIVE_MAGNITUDE_RATIO_MAX,
            "min_marginal_delta_pp": MIN_MARGINAL_DELTA_PP,
            "note": (
                "substitutive: mean |within-B-stratum A delta| < substitutive_ratio_max x |A's pooled marginal "
                "delta|. additive: not substitutive, every B-stratum delta shares the marginal delta's sign, and "
                "max/min |delta| ratio across strata <= additive_magnitude_ratio_max. interactive: neither of the "
                "above. inconclusive: |marginal delta| < min_marginal_delta_pp, or too few populated quartile "
                "combinations. Every delta (marginal and within-stratum) is computed between the lowest and "
                "highest A-bin that both have at least min_cell_n_for_delta rows -- a sparse tail bin (e.g. n=2 "
                "in a heavily skewed discrete feature) is excluded as an endpoint rather than producing a "
                "noise-driven delta."
            ),
            "min_cell_n_for_delta": MIN_CELL_N_FOR_DELTA,
        },
        "candidate_selection": (
            "Top 5 active + top 5 passive features ranked by count of DISTINCT slicers (from "
            "reports/eda/SLICE_STRATIFICATION.json's cross_dataset_summary.genuine_divergences) they diverge on, "
            "after collapsing prompt 27's consensus-redundant slicer clusters to one representative each (active: "
            "phase_label_prev_event folds into phase_label; position_group -- not itself a locked feature, only "
            "used as a slicer in prompt 24's original cells -- folds into position, Cramer's V=1.0 per "
            "feature_config.py; passive has no consensus-redundant clusters, all 4 slicers counted separately). "
            "Ties broken by total raw divergence-row count, then alphabetically. See module docstring for the "
            "exact ranked lists and the football rationale behind each pairing."
        ),
        "summary_table": summary_table,
        "pairs": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
