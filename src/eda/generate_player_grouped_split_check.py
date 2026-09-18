"""CLI entrypoint: EDA open item #1 -- player-grouped CV stress test for the
active leg's identity-leakage risk on `position`.

Context: the canonical, frozen split (scripts/pipeline/compute_canonical_split.py) is
grouped by match_id, not player_id. That is the right default for evaluating
match-to-match generalisation, but it does NOT guarantee a given player is
absent from a fold's held-out validation rows if they only appear in a
handful of matches -- a model could in principle partially memorise a
player's identity (via `position` acting as a weak proxy, or via features
that correlate strongly with a specific player's role) rather than learning
the general defensive-actions relationship. This script builds a SEPARATE,
player-grouped fold structure (reusing the existing generic
`make_grouped_folds(group_col=...)` helper, not new fold logic) restricted
to the TRAIN+VAL rows of the canonical split (the frozen TEST match set is
left untouched), verifies zero player overlap across folds, and compares
fold balance (rows / matches / players / target rate) against the canonical
match-grouped folds over the same TRAIN+VAL rows.

This does NOT replace the canonical split. It is a diagnostic: stage 06
(modelling) can use this player-grouped fold structure specifically when
stress-testing whether a model's apparent skill degrades once no fold ever
sees the same player twice -- if performance is materially unchanged, the
identity-leakage risk from `position` is not a practical concern; if it
degrades sharply, that is a real finding to carry into modelling.

Usage:
    python -m src.eda.generate_player_grouped_split_check
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.dax.models.splits import canonical_test_mask, load_canonical_split, make_grouped_folds

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_PATH = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "PLAYER_GROUPED_SPLIT_CHECK.json"

TARGET = "target_future_shot_10s"
XG_TARGET = "target_future_xg_10s"
N_SPLITS = 5
SEED = 42


def _match_grouped_fold_summary(df: pd.DataFrame) -> list[dict]:
    """Canonical match-grouped folds over the same TRAIN+VAL rows, for a
    like-for-like comparison against the new player-grouped folds."""
    assignment = load_canonical_split()
    match_ids = df["match_id"].astype(str)
    labels = match_ids.map(assignment)
    fold_re_ok = labels.str.match(r"^fold\d+$")
    sub = df.loc[fold_re_ok].copy()
    sub["fold"] = labels[fold_re_ok].str.replace("fold", "", regex=False).astype(int)
    rows = []
    for fold, g in sub.groupby("fold"):
        rows.append(
            {
                "fold": int(fold),
                "rows": int(len(g)),
                "matches": int(g["match_id"].nunique()),
                "players": int(g["player_id"].nunique()),
                "target_mean": float(g[TARGET].mean()),
                "positive_shots": int(g[TARGET].sum()),
            }
        )
    return sorted(rows, key=lambda r: r["fold"])


def _player_grouped_fold_summary(df: pd.DataFrame, folds: pd.DataFrame) -> list[dict]:
    joined = df.join(folds["fold"])
    rows = []
    for fold, g in joined.groupby("fold"):
        rows.append(
            {
                "fold": int(fold),
                "rows": int(len(g)),
                "matches": int(g["match_id"].nunique()),
                "players": int(g["player_id"].nunique()),
                "target_mean": float(g[TARGET].mean()),
                "positive_shots": int(g[TARGET].sum()),
            }
        )
    return sorted(rows, key=lambda r: r["fold"])


def _position_distribution(df: pd.DataFrame, folds: pd.DataFrame) -> list[dict]:
    joined = df.join(folds["fold"])
    out = []
    overall = joined["position"].value_counts(normalize=True).to_dict()
    for fold, g in joined.groupby("fold"):
        dist = g["position"].value_counts(normalize=True).to_dict()
        max_abs_dev = max(abs(dist.get(pos, 0.0) - overall.get(pos, 0.0)) for pos in overall)
        out.append({"fold": int(fold), "max_abs_position_share_deviation": round(float(max_abs_dev), 4)})
    return sorted(out, key=lambda r: r["fold"])


def main() -> None:
    df = pd.read_parquet(
        ACTIVE_PATH,
        columns=["match_id", "player_id", "position", TARGET, XG_TARGET],
    )
    df["match_id"] = df["match_id"].astype(str)

    test_mask = canonical_test_mask(df)
    train_val = df.loc[~test_mask].reset_index(drop=True)

    n_players_total = train_val["player_id"].nunique()

    player_folds = make_grouped_folds(train_val, target=TARGET, group_col="player_id", n_splits=N_SPLITS, seed=SEED)

    # Verify zero player overlap across folds -- the entire point of this
    # exercise. GroupKFold/StratifiedGroupKFold guarantee this by
    # construction, but we check it directly rather than trusting the
    # library silently.
    player_to_folds = player_folds.groupby("player_id")["fold"].nunique()
    leaking_players = int((player_to_folds > 1).sum())

    player_summary = _player_grouped_fold_summary(train_val, player_folds)
    match_summary = _match_grouped_fold_summary(train_val)
    position_balance = _position_distribution(train_val, player_folds)

    result = {
        "purpose": (
            "Stress-test identity-leakage risk on `position` for the active leg by building a player-disjoint "
            "fold structure (no player appears in more than one fold) and comparing fold balance against the "
            "canonical match-grouped folds over the same TRAIN+VAL rows."
        ),
        "scope": "Active leg only (player_id / position are active-dataset concepts; the passive leg is not player-indexed the same way).",
        "canonical_test_set_excluded": True,
        "n_train_val_rows": int(len(train_val)),
        "n_train_val_matches": int(train_val["match_id"].nunique()),
        "n_train_val_players": int(n_players_total),
        "n_splits": N_SPLITS,
        "seed": SEED,
        "leakage_check": {
            "players_appearing_in_more_than_one_fold": leaking_players,
            "verdict": "clean -- zero player overlap across folds" if leaking_players == 0 else "FAILED -- player overlap detected",
        },
        "player_grouped_fold_summary": player_summary,
        "match_grouped_fold_summary_same_rows": match_summary,
        "position_balance_player_grouped": position_balance,
        "interpretation": (
            "If player-grouped and match-grouped fold summaries show similar target rates and row counts (no "
            "fold collapsing or wildly skewed target_mean), the two split structures are broadly interchangeable "
            "for this dataset -- meaning the canonical match-grouped split's per-fold estimates are not being "
            "propped up by within-fold player repetition. Position balance is checked separately since `position` "
            "is the specific leakage vector named in stage 06 planning: a large max_abs_position_share_deviation "
            "in any fold means that fold's position mix diverges from the dataset average, which is the "
            "condition under which position-as-identity-proxy leakage would actually bite."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Leakage check: {result['leakage_check']['verdict']}")


if __name__ == "__main__":
    main()
