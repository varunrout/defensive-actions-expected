"""CLI entrypoint: prompt 31 -- player-level repeated-measures / split-
validity check. Every validity check so far (train/test consistency in
prompt 21, tournament-stability in prompt 24) checked whether patterns hold
across matches/tournaments. StratifiedGroupKFold (see
outputs/models/splits/match_assignment.json) already groups by match, so no
single match leaks across train/test -- it does NOT group by player. If a
small number of players contribute a disproportionate share of rows, the
same players show up in both train and test doing structurally similar
things, and a feature that's more "who is defending" than "what is the
defensive shape" would look more predictive during validation than it will
on a genuinely new player.

Active dataset only: player_defensive_actions.parquet has player_id (one
row per actual defensive action, confirmed present via direct column
check). passive_defense.parquet does NOT -- it's one row per anonymous
defender-slot per event, no player identity column at all. This is a real
scope limitation, not an oversight -- stated in the output JSON/HTML
itself, not just in this docstring.

This check is TARGET-INDEPENDENT: row concentration, per-feature ICC
against player_id, and train/test player overlap never reference
target_future_shot_10s or target_future_xg_10s at all -- the numbers are
identical whichever target a model downstream would use. Mirrored into
reports/eda_xg/ unchanged (same pattern already established for Correlation
Atlas / VIF / Slicer Redundancy in generate_correlation_vif_xg_mirror.py),
not recomputed against xg.

Step 2 method note: a one-way ICC / variance-ratio (between-player
variance / total variance) only has a natural definition for a NUMERIC
feature. For the 6 genuinely categorical locked active features
(phase_label, position, event_type, play_pattern, phase_label_prev_event,
period), variance-ratio is not well-defined -- Cramer's V between
player_id and the feature is used instead (same bias-corrected function
prompt 27 used, src.eda.correlation.cramers_v), which is also 0-1-bounded
and supports the same 0.3 threshold, but is a different statistic (chi-
square-based association, not a variance decomposition) -- each feature's
row states which method produced its value.

Usage:
    python -m src.eda.generate_player_level_validity_check
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.eda.correlation import cramers_v, correlation_ratio
from src.eda.feature_config import ACTIVE, PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "PLAYER_LEVEL_VALIDITY_CHECK.json"
SPLIT_PATH = REPO_ROOT / "outputs" / "models" / "splits" / "match_assignment.json"

ICC_THRESHOLD = 0.3
PLAYER_COL = "player_id"


def _gini(counts: np.ndarray) -> float:
    x_sorted = np.sort(counts.astype(float))
    n = len(x_sorted)
    cum = np.cumsum(x_sorted)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def row_concentration(df: pd.DataFrame) -> dict:
    counts = df[PLAYER_COL].value_counts().sort_values(ascending=False)
    total = int(counts.sum())
    n_players = len(counts)

    lorenz = []
    for pct in (10, 25, 50, 75, 100):
        k = max(1, round(n_players * pct / 100))
        cum_rows = int(counts.iloc[:k].sum())
        lorenz.append({"pct_of_players": pct, "n_players": k, "cumulative_row_share": round(cum_rows / total, 4)})

    return {
        "n_rows": total,
        "n_distinct_players": n_players,
        "top_10_players_row_share": round(float(counts.iloc[:10].sum() / total), 4),
        "top_25_players_row_share": round(float(counts.iloc[:25].sum() / total), 4),
        "top_50_players_row_share": round(float(counts.iloc[:50].sum() / total), 4),
        "gini_coefficient": round(_gini(counts.values), 4),
        "max_rows_single_player": int(counts.max()),
        "median_rows_per_player": float(counts.median()),
        "lorenz_curve_by_player_rank": lorenz,
    }


def feature_icc_ranking(df: pd.DataFrame) -> list[dict]:
    rows = []
    for feature_type in ("continuous", "discrete", "boolean"):
        for feature in ACTIVE[feature_type]:
            sub = df[[PLAYER_COL, feature]].dropna()
            values = sub[feature].astype(float) if feature_type != "boolean" else sub[feature].astype(bool).astype(float)
            value = correlation_ratio(sub[PLAYER_COL], values) ** 2  # eta -> eta^2 = variance ratio (ICC proxy)
            rows.append({"feature": feature, "feature_type": feature_type, "method": "variance_ratio (eta^2 of feature vs player_id groups)", "value": round(value, 4)})
    for feature in ACTIVE["categorical"]:
        sub = df[[PLAYER_COL, feature]].dropna()
        value = cramers_v(sub[PLAYER_COL], sub[feature])
        rows.append({"feature": feature, "feature_type": "categorical", "method": "cramers_v (player_id vs feature, bias-corrected)", "value": round(float(value), 4) if not np.isnan(value) else None})

    rows.sort(key=lambda r: (r["value"] is None, -(r["value"] or 0)))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r["player_identity_flavoured"] = (r["value"] or 0) > ICC_THRESHOLD
    return rows


def train_test_player_overlap(df: pd.DataFrame) -> dict:
    assignment = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    match_id_str = df["match_id"].astype(str)
    split = match_id_str.map(assignment)
    is_test = split == "test"
    is_trainval = split.notna() & ~is_test

    test_players = set(df.loc[is_test, PLAYER_COL])
    trainval_players = set(df.loc[is_trainval, PLAYER_COL])
    overlap = test_players & trainval_players
    overlap_pct = round(100 * len(overlap) / len(test_players), 2) if test_players else None

    return {
        "n_test_matches": int(match_id_str[is_test].nunique()),
        "n_trainval_matches": int(match_id_str[is_trainval].nunique()),
        "n_test_players": len(test_players),
        "n_trainval_players": len(trainval_players),
        "n_overlap_players": len(overlap),
        "overlap_pct_of_test_players": overlap_pct,
        "note": (
            "Some overlap is expected and not inherently a problem -- players play in multiple matches across "
            "this 115-match, 2-tournament dataset. The percentage itself, not an assumption about it, is what "
            "matters for Step 3's risk-pair cross-reference."
        ),
    }


def main() -> None:
    df = pd.read_parquet(REPO_ROOT / ACTIVE["parquet_path"])

    # Confirmed directly, not assumed: passive_defense.parquet has no
    # player identity column at all.
    passive_schema_cols = set(pq.ParquetFile(REPO_ROOT / PASSIVE["parquet_path"]).schema.names)
    assert not any("player" in c.lower() for c in passive_schema_cols), (
        f"Expected no player-identity column in passive_defense.parquet, found candidates: "
        f"{[c for c in passive_schema_cols if 'player' in c.lower()]} -- scope limitation claim needs updating."
    )

    concentration = row_concentration(df)
    icc_ranking = feature_icc_ranking(df)
    overlap = train_test_player_overlap(df)

    overlap_pct = overlap["overlap_pct_of_test_players"] or 0
    high_icc_features = [r for r in icc_ranking if r["player_identity_flavoured"]]
    risk_pairs = []
    for r in high_icc_features:
        if overlap_pct >= 50:
            risk_level = "high" if (r["value"] or 0) > 0.5 else "moderate"
        else:
            risk_level = "low"
        risk_pairs.append({
            "feature": r["feature"],
            "icc_or_cramers_v": r["value"],
            "method": r["method"],
            "train_test_player_overlap_pct": overlap_pct,
            "risk_level": risk_level,
            "reason": (
                f"{r['feature']} is player-identity-flavoured ({r['value']} > {ICC_THRESHOLD}) AND "
                f"{overlap_pct}% of test-set players also appear in train+val -- the combination the prompt "
                f"asks to name explicitly, not either fact alone."
                if overlap_pct >= 50 else
                f"{r['feature']} is player-identity-flavoured ({r['value']} > {ICC_THRESHOLD}), but train/test "
                f"player overlap is only {overlap_pct}% -- the identity signal is less likely to leak across the "
                "split since most test players aren't seen in training."
            ),
        })

    output = {
        "dataset": "active",
        "scope_limitation": (
            "This check is active-dataset only. passive_defense.parquet has no player identity column -- it's "
            "one row per anonymous defender-slot per on-ball event, not per identified player -- so player-level "
            "ICC and train/test player overlap cannot be computed on the passive side. This is a real scope "
            "limitation of the data, not an oversight: there is no workaround that recovers player identity from "
            "an anonymous defender slot."
        ),
        "target_independence_note": (
            "Row concentration, per-feature ICC/Cramer's V against player_id, and train/test player overlap "
            "never reference target_future_shot_10s or target_future_xg_10s -- these numbers are identical "
            "whichever target a downstream model uses. Mirrored unchanged into reports/eda_xg/, not recomputed."
        ),
        "row_concentration": concentration,
        "icc_threshold": ICC_THRESHOLD,
        "icc_methodology": (
            "Variance-ratio (eta^2, src.eda.correlation.correlation_ratio squared) for continuous/discrete/"
            "boolean features (28 of 34) -- the standard one-way ICC proxy, between-player variance / total "
            "variance, treating player_id as the grouping factor. Cramer's V (bias-corrected, "
            "src.eda.correlation.cramers_v) for the 6 genuinely categorical features, since variance-ratio has no "
            "natural definition for a non-numeric feature -- same 0-1 scale and the same 0.3 threshold, but a "
            "different statistic (chi-square association, not a variance decomposition); each row states which "
            "method produced its value."
        ),
        "feature_icc_ranking": icc_ranking,
        "train_test_player_overlap": overlap,
        "risk_pairs": risk_pairs,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print(f"n_rows={concentration['n_rows']}, n_players={concentration['n_distinct_players']}, gini={concentration['gini_coefficient']}")
    print(f"top10/25/50 row share: {concentration['top_10_players_row_share']}/{concentration['top_25_players_row_share']}/{concentration['top_50_players_row_share']}")
    print(f"train/test player overlap: {overlap['overlap_pct_of_test_players']}%")
    print(f"\nICC ranking (top 10 of 34):")
    for r in icc_ranking[:10]:
        print(f"  {r['rank']}. {r['feature']} ({r['feature_type']}): {r['value']} [{r['method']}]{' -- FLAGGED' if r['player_identity_flavoured'] else ''}")
    print(f"\n{len(risk_pairs)} risk pair(s):")
    for rp in risk_pairs:
        print(f"  {rp['feature']}: {rp['risk_level']} -- {rp['reason']}")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
