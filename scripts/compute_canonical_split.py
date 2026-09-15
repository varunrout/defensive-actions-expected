"""Compute the canonical, frozen match-grouped split (prompt 13) and write
outputs/models/splits/match_assignment.json.

Run once. The resulting assignment is meant to be loaded (via
dax.models.splits.load_canonical_split), never regenerated per training run
-- regenerating it would silently change which matches are held out, which
defeats the point of freezing a shared, comparable split across the active
and passive legs.

Usage:
    python scripts/compute_canonical_split.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "outputs" / "models" / "splits" / "match_assignment.json"

ACTIVE_PATH = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
PASSIVE_PATH = REPO_ROOT / "data" / "features" / "passive_defense.parquet"
TARGET = "target_future_shot_10s"

TEST_SIZE = 0.20
SEED = 42
N_SPLITS = 5


def _per_match_rate(path: Path) -> pd.Series:
    df = pd.read_parquet(path, columns=["match_id", TARGET])
    df["match_id"] = df["match_id"].astype(str)
    return df.groupby("match_id")[TARGET].mean()


def main() -> None:
    active_rate = _per_match_rate(ACTIVE_PATH)
    passive_rate = _per_match_rate(PASSIVE_PATH)

    matches = sorted(set(active_rate.index) | set(passive_rate.index))
    if set(active_rate.index) != set(passive_rate.index):
        raise ValueError(
            "Active and passive datasets cover different match_id sets -- "
            "the shared split assumes both legs see the same matches."
        )

    match_df = pd.DataFrame({"match_id": matches})
    match_df["active_rank"] = match_df["match_id"].map(active_rate).rank(pct=True)
    match_df["passive_rank"] = match_df["match_id"].map(passive_rate).rank(pct=True)
    match_df["combined_rank"] = (match_df["active_rank"] + match_df["passive_rank"]) / 2
    match_df["rank_quartile"] = pd.qcut(match_df["combined_rank"], 4, labels=False, duplicates="drop")

    train_val_ids, test_ids = train_test_split(
        match_df["match_id"].to_numpy(),
        test_size=TEST_SIZE,
        stratify=match_df["rank_quartile"].to_numpy(),
        random_state=SEED,
    )

    print(f"TEST: {len(test_ids)} matches, TRAIN+VAL: {len(train_val_ids)} matches")

    # Build a row-level frame over the TRAIN+VAL matches only, for the
    # StratifiedGroupKFold split -- one row per (match_id, active_row)
    # is enough since the fold grouping is purely by match_id; use the
    # active dataset's row-level target for the stratification (passive is
    # ~30x larger and would dominate row counts without changing which
    # matches land in which fold, since grouping is match-level).
    active_df = pd.read_parquet(ACTIVE_PATH, columns=["match_id", TARGET])
    active_df["match_id"] = active_df["match_id"].astype(str)
    train_val_rows = active_df[active_df["match_id"].isin(train_val_ids)].reset_index(drop=True)

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    match_to_fold: dict[str, int] = {}
    for fold_idx, (_, val_idx) in enumerate(
        skf.split(train_val_rows, train_val_rows[TARGET], groups=train_val_rows["match_id"])
    ):
        for mid in train_val_rows.iloc[val_idx]["match_id"].unique():
            match_to_fold[mid] = fold_idx

    missing = set(train_val_ids) - set(match_to_fold.keys())
    if missing:
        raise ValueError(f"{len(missing)} TRAIN+VAL matches were not assigned to any fold: {sorted(missing)}")

    assignment: dict[str, str] = {}
    for mid in test_ids:
        assignment[str(mid)] = "test"
    for mid, fold_idx in match_to_fold.items():
        assignment[str(mid)] = f"fold{fold_idx}"

    if set(assignment.keys()) != set(matches):
        raise ValueError("Assignment does not cover exactly the 115 matches seen in the data.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(assignment, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(assignment)} matches)")

    # Sanity-check summary, matching the acceptance-check shape (not exact
    # numbers) requested in the prompt.
    def _summary(path: Path, label: str) -> None:
        df = pd.read_parquet(path, columns=["match_id", TARGET])
        df["match_id"] = df["match_id"].astype(str)
        df["split"] = df["match_id"].map(assignment)
        test_rows = df[df["split"] == "test"]
        print(f"\n{label} TEST: {len(test_rows):,} rows, {test_rows[TARGET].mean() * 100:.2f}% shot rate")
        for fold in sorted(f for f in df["split"].unique() if f != "test"):
            fold_rows = df[df["split"] == fold]
            n_matches = fold_rows["match_id"].nunique()
            print(f"  {fold}: {n_matches} matches, {len(fold_rows):,} rows, {fold_rows[TARGET].mean() * 100:.2f}% shot rate")

    _summary(ACTIVE_PATH, "active")
    _summary(PASSIVE_PATH, "passive")


if __name__ == "__main__":
    main()
